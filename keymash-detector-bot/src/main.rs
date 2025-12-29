use std::{
    collections::HashMap,
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use serde::{Deserialize, Serialize};
use twitch_highway::{
    TwitchAPI,
    chat::ChatAPI,
    eventsub::{
        EventSubAPI, SubscriptionType,
        events::chat::ChannelChatMessage,
        websocket::{self, extract},
    },
    types::{BroadcasterId, UserId},
};
use twitch_oauth_token::{AccessToken, ClientId, ClientSecret, RefreshToken, TwitchOauth, scope};

// this is public
const TWITCH_CLIENT_ID: &str = include_str!("CLIENT_ID").trim_ascii_end();
const CONFIG_FILE_PATH: &str = "bot-config.toml";
// twitch_highway does not forward twitch keepalives, so minimize them
const TWITCH_EVENTSUB_WS_URL: &str =
    "wss://eventsub.wss.twitch.tv/ws?keepalive_timeout_seconds=600";

struct RunState {
    api_app: TwitchAPI,
    api_user: TwitchAPI,
    me_user: UserId,
    rooms: HashMap<BroadcasterId, (RoomConfig, AtomicBool)>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::fmt()
        .with_env_filter("info,twitch_highway=debug,keymash_detector_bot=trace")
        .init();

    // TODO: refresh tokens as needed
    let run_state = init().await?;

    {
        use websocket::routes::*;
        websocket::client(
            TWITCH_EVENTSUB_WS_URL,
            websocket::Router::new()
                .route(welcome(on_welcome))
                .route(channel_chat_message(on_message))
                .with_state(Arc::new(run_state)),
        )
        .await?;
    }

    Ok(())
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
struct Config {
    client: ClientConfig,
    bot_user: UserConfig,
    rooms: HashMap<BroadcasterId, RoomConfig>,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
struct ClientConfig {
    id: ClientId,
    secret: ClientSecret,
    access_token: AccessToken,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
struct UserConfig {
    id: String,
    refresh_token: RefreshToken,
    access_token: AccessToken,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RoomConfig {
    has_bot_auth: bool,
}

async fn init() -> Result<RunState, Box<dyn std::error::Error>> {
    #[derive(Deserialize)]
    struct ValidateAppToken {
        client_id: ClientId,
        // expires_in: u64,
    }

    let config: Config = toml::from_str(&fs::read_to_string(CONFIG_FILE_PATH)?)?;

    let client_id = ClientId::from(TWITCH_CLIENT_ID);
    assert!(config.client.id == client_id);
    let oauth = TwitchOauth::from_credentials(client_id.clone(), config.client.secret.clone());

    let app_valid = oauth
        .validate_access_token(&config.client.access_token)
        .await;
    let app_token = match app_valid {
        Ok(app_valid) => {
            tracing::debug!("stored app access token was valid");
            let app_valid: ValidateAppToken = app_valid.json().await?;
            assert!(app_valid.client_id == client_id);
            config.client.access_token
        }
        Err(e) => {
            tracing::info!("stored app access token was invalid; re-authing...");
            tracing::debug!("validate response: {:?}", e);
            let app_token = oauth.app_access_token().await?.app_token();
            app_token.await?.access_token
        }
    };

    let check_user_info = |user_valid: twitch_oauth_token::ValidateToken| {
        assert!(user_valid.client_id == client_id);
        assert!(user_valid.user_id == config.bot_user.id);
        assert!(user_valid.scopes.contains(&scope::Scope::UserBot));
    };

    let user_valid = oauth
        .validate_access_token(&config.bot_user.access_token)
        .await;
    let (user_refresh_token, user_token) = match user_valid {
        Ok(user_valid) => {
            tracing::debug!("stored user access token was valid");
            check_user_info(user_valid.validate_token().await?);
            (config.bot_user.refresh_token, config.bot_user.access_token)
        }
        Err(e) => {
            tracing::info!("stored user access token was invalid; refreshing...");
            tracing::debug!("validate response: {:?}", e);
            let user_token = oauth.refresh_access_token(config.bot_user.refresh_token.clone());
            let user_token = user_token.await?.user_token().await?;
            if user_token.refresh_token != config.bot_user.refresh_token {
                tracing::info!("refresh token changed")
            }
            let user_valid = oauth.validate_access_token(&user_token.access_token);
            check_user_info(user_valid.await?.validate_token().await?);
            (user_token.refresh_token, user_token.access_token)
        }
    };

    let new_config = Config {
        client: ClientConfig {
            access_token: app_token,
            ..config.client
        },
        bot_user: UserConfig {
            refresh_token: user_refresh_token,
            access_token: user_token,
            ..config.bot_user
        },
        rooms: config.rooms,
    };
    fs::write(CONFIG_FILE_PATH, toml::to_string(&new_config)?)?;

    Ok(RunState {
        api_app: TwitchAPI::new(new_config.client.access_token, client_id.clone()),
        api_user: TwitchAPI::new(new_config.bot_user.access_token, client_id),
        me_user: UserId::from(new_config.bot_user.id),
        rooms: new_config
            .rooms
            .into_iter()
            .map(|(k, v)| (k, (v, AtomicBool::new(false))))
            .collect(),
    })
}

async fn on_welcome(
    extract::State(run_state): extract::State<Arc<RunState>>,
    extract::Session(session): extract::Session,
) {
    for room_user in run_state.rooms.keys() {
        tracing::info!("subscribing to room {}", room_user);
        let response = run_state
            .api_user
            .websocket_subscription(SubscriptionType::ChannelChatMessage, session.id.clone())
            .broadcaster_user_id(room_user.clone())
            .user_id(run_state.me_user.clone())
            .json()
            .await;
        if let Err(e) = response {
            tracing::error!("failed to subscribe to room {}: {:?}", room_user, e);
        }
    }
}

async fn on_message(
    extract::State(run_state): extract::State<Arc<RunState>>,
    extract::Event(event): extract::Event<ChannelChatMessage>,
) {
    let room_id = &event.broadcaster_user_id;
    let room_state = &run_state.rooms[room_id];

    if event.chatter_user_id == run_state.me_user {
        tracing::debug!("ignored self-chat in room {}", room_id);
        return;
    }
    tracing::debug!("recv chat in room {}: {}", room_id, event.message.text);

    let duplicate_bypass_state = room_state.1.fetch_not(Ordering::SeqCst);
    let mut msg = "auto reply spam test".to_string();
    if duplicate_bypass_state {
        // from 7tv "Bypass Duplicate Message Check"
        // chatterino apparently uses `'\u{E0000}'`
        msg.push_str(" \u{34f}");
    }
    let api = if room_state.0.has_bot_auth {
        &run_state.api_app
    } else {
        &run_state.api_user
    };

    let response = api
        .send_chat_message(room_id, &run_state.me_user, &msg)
        .reply_parent_message_id(&event.message_id)
        .json()
        .await;

    if let Ok(r) = &response
        && let [r] = &r.data[..]
        && r.is_sent
    {
        tracing::debug!("chat reply sent");
    } else {
        tracing::warn!("failed to send chat: {:?}", response);
    }
}

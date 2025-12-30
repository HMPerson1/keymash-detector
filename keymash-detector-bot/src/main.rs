use std::{collections::HashMap, fs, ops::Deref, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use tokio::{join, sync::mpsc, time::Instant};
use twitch_highway::{
    TwitchAPI,
    chat::{ChatAPI, SendChatMessageResponse},
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
// 20msg/30sec, plus some safety margin
const MIN_CHAT_WAIT: Duration = Duration::from_millis(1_510);

struct GlobalState {
    api_app: TwitchAPI,
    api_user: TwitchAPI,
    me_user: UserId,
}

struct RunState {
    globals: Arc<GlobalState>,
    rooms: Vec<BroadcasterId>,
    chat_sender: mpsc::Sender<SendChatReq>,
}

impl Deref for RunState {
    type Target = GlobalState;

    fn deref(&self) -> &Self::Target {
        self.globals.deref()
    }
}

struct SendChatReq {
    room_id: BroadcasterId,
    msg: String,
    reply_to_id: String,
}

struct RoomRunState {
    has_bot_auth: bool,
    duplicate_bypass_state: bool,
}

impl From<RoomConfig> for RoomRunState {
    fn from(value: RoomConfig) -> Self {
        Self {
            has_bot_auth: value.has_bot_auth,
            duplicate_bypass_state: false,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::fmt()
        .with_env_filter("info,twitch_highway=debug,keymash_detector_bot=trace")
        .init();

    // TODO: refresh tokens as needed
    let config = init().await?;

    let globals = Arc::new(GlobalState {
        api_app: TwitchAPI::new(config.client.access_token, config.client.id.clone()),
        api_user: TwitchAPI::new(config.bot_user.access_token, config.client.id),
        me_user: UserId::from(config.bot_user.id),
    });

    let (tx, rx) = mpsc::channel(50);

    let run_state = RunState {
        globals: globals.clone(),
        rooms: config.rooms.keys().cloned().collect(),
        chat_sender: tx,
    };

    let sender_task = tokio::spawn(send_chat_loop(globals, config.rooms, rx));

    let reader_task = tokio::spawn({
        use websocket::routes::*;
        websocket::client(
            TWITCH_EVENTSUB_WS_URL,
            websocket::Router::new()
                .route(welcome(on_welcome))
                .route(channel_chat_message(on_message))
                .with_state(Arc::new(run_state)),
        )
        .into_future()
    });

    let _ = join!(sender_task, reader_task);

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

async fn init() -> Result<Config, Box<dyn std::error::Error>> {
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

    Ok(new_config)
}

async fn on_welcome(
    extract::State(run_state): extract::State<Arc<RunState>>,
    extract::Session(session): extract::Session,
) {
    for room_user in &run_state.rooms {
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
    let room_id = event.broadcaster_user_id.clone();

    let msg = {
        if event.chatter_user_id == run_state.me_user {
            tracing::debug!("ignored self-chat in room {}", room_id);
            return;
        } else {
            tracing::debug!("recv chat in room {}: {}", room_id, event.message.text);
            "auto reply spam test".to_string()
        }
    };

    let enqueue_res = run_state.chat_sender.try_send({
        SendChatReq {
            room_id,
            msg,
            reply_to_id: event.message_id,
        }
    });
    if let Err(e) = enqueue_res {
        tracing::error!("failed to enqueue chat send: {}", e);
    }
}

async fn send_chat_loop(
    run_state: Arc<GlobalState>,
    config_rooms: HashMap<BroadcasterId, RoomConfig>,
    mut rx: mpsc::Receiver<SendChatReq>,
) {
    let mut rooms: HashMap<_, RoomRunState> = config_rooms
        .into_iter()
        .map(|(k, v)| (k, v.into()))
        .collect();
    let mut last_chat_send_attempt = Instant::now() - Duration::from_hours(1);

    while let Some(mut req) = rx.recv().await {
        let room_state = rooms.get_mut(&req.room_id).unwrap();

        if room_state.duplicate_bypass_state {
            // from 7tv "Bypass Duplicate Message Check"
            // chatterino apparently uses `'\u{E0000}'`
            req.msg.push_str(" \u{34f}");
        }

        let mut do_send_chat = async |api: &TwitchAPI| {
            tokio::time::sleep_until(last_chat_send_attempt + MIN_CHAT_WAIT).await;
            last_chat_send_attempt = Instant::now();
            api.send_chat_message(&req.room_id, &run_state.me_user, &req.msg)
                .reply_parent_message_id(&req.reply_to_id)
                .json()
                .await
        };

        if room_state.has_bot_auth {
            let response = do_send_chat(&run_state.api_app).await;

            if is_send_chat_sent(&response) {
                room_state.duplicate_bypass_state = !room_state.duplicate_bypass_state;
                tracing::debug!("chat reply sent as bot");
                continue;
            }
            tracing::warn!("failed to send chat as bot: {:?}", &response);
            if !is_send_chat_failed_bot_auth(&response) {
                // don't retry for other errors
                continue;
            }
            tracing::warn!("unsetting has_bot_auth and retrying...");
            room_state.has_bot_auth = false;
            // fallthrough to `has_bot_auth == false`
        }

        let response = do_send_chat(&run_state.api_user).await;

        if is_send_chat_sent(&response) {
            room_state.duplicate_bypass_state = !room_state.duplicate_bypass_state;
            tracing::debug!("chat reply sent");
        } else {
            tracing::warn!("failed to send chat: {:?}", &response);
        }
    }
}

fn is_send_chat_sent(response: &Result<SendChatMessageResponse, twitch_highway::Error>) -> bool {
    if let Ok(r) = response
        && let [r] = &r.data[..]
        && r.is_sent
    {
        true
    } else {
        false
    }
}

fn is_send_chat_failed_bot_auth(
    response: &Result<SendChatMessageResponse, twitch_highway::Error>,
) -> bool {
    if let Err(e) = &response
        && e.is_api()
        && let Some(err_msg) = e.message()
        // kinda jank but meh
        && err_msg.contains("channel:bot scope")
        && err_msg.contains("moderator")
    {
        true
    } else {
        false
    }
}

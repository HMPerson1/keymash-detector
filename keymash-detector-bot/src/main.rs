use std::{collections::HashMap, fs, ops::Deref, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use tokio::{
    join,
    sync::{mpsc, watch},
    time::Instant,
};
use tracing::Instrument;
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

const HTTP_CLIENT_UA: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
// this is public
const TWITCH_CLIENT_ID: &str = include_str!("CLIENT_ID").trim_ascii_end();
const SECRETS_FILE_PATH: &str = "bot-secrets.toml";
const CONFIG_FILE_PATH: &str = "bot-config.toml";
// twitch_highway does not forward twitch keepalives, so minimize them
const TWITCH_EVENTSUB_WS_URL: &str =
    "wss://eventsub.wss.twitch.tv/ws?keepalive_timeout_seconds=600";
// 20msg/30sec, plus some safety margin
const MIN_CHAT_WAIT: Duration = Duration::from_millis(1_510);

#[derive(Clone)]
struct GlobalState {
    client_id: ClientId,
    app_token: watch::Receiver<AccessToken>,
    user_token: watch::Receiver<AccessToken>,
    http_client: reqwest::Client,
}

impl GlobalState {
    fn api_app(&self) -> TwitchAPI {
        TwitchAPI::with_client(
            self.app_token.borrow().clone(),
            self.client_id.clone(),
            self.http_client.clone(),
        )
    }
    fn api_user(&self) -> TwitchAPI {
        TwitchAPI::with_client(
            self.user_token.borrow().clone(),
            self.client_id.clone(),
            self.http_client.clone(),
        )
    }
}

struct RunState {
    globals: GlobalState,
    me_user: UserId,
    rooms: HashMap<BroadcasterId, tracing::Span>,
    chat_sender: mpsc::Sender<SendChatReq>,
}

impl Deref for RunState {
    type Target = GlobalState;

    fn deref(&self) -> &Self::Target {
        &self.globals
    }
}

struct SendChatReq {
    span: tracing::Span,
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

    let secrets = validate_secrets().await?;
    let me_user = UserId::from(secrets.bot_user.id);

    let (app_token_tx, app_token_rx) = watch::channel(secrets.client.access_token);
    let (user_token_tx, user_token_rx) = watch::channel(secrets.bot_user.access_token);
    let validate_task = tokio::spawn(token_validate_loop(app_token_tx, user_token_tx));

    let globals = GlobalState {
        client_id: secrets.client.id,
        app_token: app_token_rx,
        user_token: user_token_rx,
        http_client: asknothingx2_util::api::preset::rest_api(HTTP_CLIENT_UA).build_client()?,
    };

    let config: Config = toml::from_str(&fs::read_to_string(CONFIG_FILE_PATH)?)?;
    let rooms: HashMap<_, _> = config
        .0
        .keys()
        .map(|k| (k.clone(), tracing::info_span!("room", id = &**k)))
        .collect();

    let (tx, rx) = mpsc::channel(50);

    let sender_task = tokio::spawn(send_chat_loop(
        globals.clone(),
        me_user.clone(),
        config.0,
        rx,
    ));

    let run_state = RunState {
        globals,
        me_user,
        rooms,
        chat_sender: tx,
    };

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

    let _ = join!(validate_task, sender_task, reader_task);

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct Secrets {
    client: ClientSecrets,
    bot_user: UserSecrets,
}

#[derive(Serialize, Deserialize)]
struct ClientSecrets {
    id: ClientId,
    secret: ClientSecret,
    access_token: AccessToken,
}

#[derive(Serialize, Deserialize)]
struct UserSecrets {
    id: String,
    refresh_token: RefreshToken,
    access_token: AccessToken,
}

#[derive(Serialize, Deserialize)]
struct Config(HashMap<BroadcasterId, RoomConfig>);

#[derive(Serialize, Deserialize)]
struct RoomConfig {
    has_bot_auth: bool,
}

#[tracing::instrument(skip_all)]
async fn token_validate_loop(
    app_token_out: watch::Sender<AccessToken>,
    user_token_out: watch::Sender<AccessToken>,
) {
    let mut interval = tokio::time::interval(Duration::from_hours(1));
    // skip initial tick since `main` validates secrets just before spawning this
    interval.tick().await;

    loop {
        interval.tick().await;

        // panic the whole process if we can't get access tokens
        let secrets = validate_secrets().await.expect("bad secrets");

        let send_res = app_token_out.send(secrets.client.access_token);
        if send_res.is_err() {
            break;
        }
        let send_res = user_token_out.send(secrets.bot_user.access_token);
        if send_res.is_err() {
            break;
        }
    }
    tracing::error!("exiting loop due to closed channel");
}

async fn validate_secrets() -> Result<Secrets, Box<dyn std::error::Error>> {
    tracing::info!("validating secrets...");
    let secrets: Secrets = toml::from_str(&fs::read_to_string(SECRETS_FILE_PATH)?)?;

    let client_id = ClientId::from(TWITCH_CLIENT_ID);
    assert!(secrets.client.id == client_id);
    let oauth = TwitchOauth::from_credentials(client_id.clone(), secrets.client.secret.clone());

    let app_token = validate_app_token(&client_id, &oauth, secrets.client.access_token).await?;

    let bot_user_id = secrets.bot_user.id.clone();
    let (user_refresh_token, user_token) =
        validate_user_token(&client_id, &oauth, secrets.bot_user).await?;

    let new_secrets = Secrets {
        client: ClientSecrets {
            access_token: app_token,
            ..secrets.client
        },
        bot_user: UserSecrets {
            id: bot_user_id,
            refresh_token: user_refresh_token,
            access_token: user_token,
        },
    };
    fs::write(SECRETS_FILE_PATH, toml::to_string(&new_secrets)?)?;

    Ok(new_secrets)
}

#[tracing::instrument(skip_all)]
async fn validate_app_token(
    client_id: &ClientId,
    oauth: &TwitchOauth,
    app_token: AccessToken,
) -> Result<AccessToken, Box<dyn std::error::Error>> {
    #[derive(Deserialize)]
    struct ValidateAppToken {
        client_id: ClientId,
        expires_in: u64,
    }
    let app_valid = oauth.validate_access_token(&app_token).await;
    match app_valid {
        Ok(app_valid) => {
            tracing::debug!("stored access token was valid");
            let app_valid: ValidateAppToken = app_valid.json().await?;
            assert!(app_valid.client_id == *client_id);
            let expires_in = Duration::from_secs(app_valid.expires_in);
            if expires_in > Duration::from_hours(4) {
                return Ok(app_token);
            }
            tracing::info!("access token expires soon: {:?}", expires_in);
        }
        Err(e) => {
            tracing::info!("stored access token was invalid");
            tracing::debug!("validate response: {:?}", e);
        }
    }
    tracing::info!("reauthing...");
    let app_token = oauth.app_access_token().await?.app_token();
    Ok(app_token.await?.access_token)
}

#[tracing::instrument(skip_all)]
async fn validate_user_token(
    client_id: &ClientId,
    oauth: &TwitchOauth,
    config_user: UserSecrets,
) -> Result<(RefreshToken, AccessToken), Box<dyn std::error::Error>> {
    let check_user_info = |user_valid: twitch_oauth_token::ValidateToken| {
        assert!(user_valid.client_id == *client_id);
        assert!(user_valid.user_id == config_user.id);
        assert!(user_valid.scopes.contains(&scope::Scope::UserBot));
    };
    let user_valid = oauth.validate_access_token(&config_user.access_token).await;
    match user_valid {
        Ok(user_valid) => {
            tracing::debug!("stored access token was valid");
            let user_valid = user_valid.validate_token().await?;
            let expires_in = Duration::from_secs(user_valid.expires_in);
            check_user_info(user_valid);
            if expires_in > Duration::from_mins(90) {
                return Ok((config_user.refresh_token, config_user.access_token));
            }
            tracing::info!("access token expires soon: {:?}", expires_in);
        }
        Err(e) => {
            tracing::info!("stored access token was invalid");
            tracing::debug!("validate response: {:?}", e);
        }
    };
    tracing::info!("refreshing...");
    let user_token = oauth.refresh_access_token(config_user.refresh_token.clone());
    let user_token = user_token.await?.user_token().await?;
    if user_token.refresh_token != config_user.refresh_token {
        tracing::info!("refresh token changed")
    }
    let user_valid = oauth.validate_access_token(&user_token.access_token);
    check_user_info(user_valid.await?.validate_token().await?);
    return Ok((user_token.refresh_token, user_token.access_token));
}

async fn on_welcome(
    extract::State(run_state): extract::State<Arc<RunState>>,
    extract::Session(session): extract::Session,
) {
    for (room_id, span) in &run_state.rooms {
        async {
            tracing::info!("subscribing...");
            let response = run_state
                .api_user()
                .websocket_subscription(SubscriptionType::ChannelChatMessage, session.id.clone())
                .broadcaster_user_id(room_id.clone())
                .user_id(run_state.me_user.clone())
                .json()
                .await;
            if let Err(e) = response {
                tracing::error!("failed: {:?}", e);
            }
        }
        .instrument(tracing::info_span!(parent: span, "join"))
        .await
    }
}

async fn on_message(
    extract::State(run_state): extract::State<Arc<RunState>>,
    extract::Event(event): extract::Event<ChannelChatMessage>,
) {
    let room_id = event.broadcaster_user_id.clone();
    let room_span = &run_state.rooms[&room_id];
    {
        let entered_span = tracing::info_span!(parent: room_span, "recv").entered();

        let msg = {
            if event.chatter_user_id == run_state.me_user {
                tracing::debug!("ignored self-chat");
                return;
            } else {
                tracing::debug!("chat: {}", event.message.text);
                "auto reply spam test".to_string()
            }
        };

        let span = tracing::info_span!(parent: room_span, "send");
        span.follows_from(&entered_span);
        let enqueue_res = run_state.chat_sender.try_send(SendChatReq {
            span,
            room_id,
            msg,
            reply_to_id: event.message_id,
        });
        if let Err(e) = enqueue_res {
            tracing::error!("failed to enqueue chat send: {}", e);
        }
    }
}

async fn send_chat_loop(
    run_state: GlobalState,
    me_user: UserId,
    config_rooms: HashMap<BroadcasterId, RoomConfig>,
    mut rx: mpsc::Receiver<SendChatReq>,
) {
    let mut rooms: HashMap<_, RoomRunState> = config_rooms
        .into_iter()
        .map(|(k, v)| (k, v.into()))
        .collect();
    let mut last_chat_send_attempt = Instant::now() - Duration::from_hours(1);

    while let Some(mut req) = rx.recv().await {
        async {
            let room_state = rooms.get_mut(&req.room_id).unwrap();

            if room_state.duplicate_bypass_state {
                // from 7tv "Bypass Duplicate Message Check"
                // chatterino apparently uses `'\u{E0000}'`
                req.msg.push_str(" \u{34f}");
            }

            let mut do_send_chat = async |api: &TwitchAPI| {
                tokio::time::sleep_until(last_chat_send_attempt + MIN_CHAT_WAIT).await;
                last_chat_send_attempt = Instant::now();
                api.send_chat_message(&req.room_id, &me_user, &req.msg)
                    .reply_parent_message_id(&req.reply_to_id)
                    .json()
                    .await
            };

            if room_state.has_bot_auth {
                let response = do_send_chat(&run_state.api_app()).await;

                if is_send_chat_sent(&response) {
                    room_state.duplicate_bypass_state = !room_state.duplicate_bypass_state;
                    tracing::debug!("chat reply sent as bot");
                    return;
                }
                tracing::warn!("failed to send chat as bot: {:?}", &response);
                if !is_send_chat_failed_bot_auth(&response) {
                    // don't retry for other errors
                    return;
                }
                tracing::warn!("unsetting has_bot_auth and retrying...");
                room_state.has_bot_auth = false;
                // fallthrough to `has_bot_auth == false`
            }

            let response = do_send_chat(&run_state.api_user()).await;

            if is_send_chat_sent(&response) {
                room_state.duplicate_bypass_state = !room_state.duplicate_bypass_state;
                tracing::debug!("chat reply sent");
            } else {
                tracing::warn!("failed to send chat: {:?}", &response);
            }
        }
        .instrument(req.span)
        .await
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

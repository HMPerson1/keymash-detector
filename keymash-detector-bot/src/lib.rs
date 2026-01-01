use twitch_highway::eventsub::events::chat;

use keymash_detector_core::*;

pub fn has_keymash(thresh: f64, message: chat::Message) -> bool {
    for frag in message.fragments {
        if !matches!(frag.kind, chat::FragmentType::Text) {
            continue;
        }

        // TODO: spanish keyboards have an <ñ> key
        let mut txt = frag.text.into_bytes();
        txt.retain(|&c| c == b' ' || c.is_ascii_graphic());

        let words: Vec<_> = txt.split(|&c| c == b' ').collect();
        for i in 0..words.len() {
            let mut word = words[i].to_owned();
            if word.len() < 6 {
                // try joining with next word
                word.push(b' ');
                word.extend_from_slice(words.get(i + 1).cloned().unwrap_or(&[]));
                if word.len() < 7 {
                    continue;
                }
            }

            let txt = preprocess(&word);
            tracing::trace!("testing: {}", str::from_utf8(&txt).unwrap());
            let llr = fit_keymash_model(&txt) - eval_english_model(&txt);
            tracing::trace!("llr: {}", llr);
            if llr > thresh {
                return true;
            }
        }
    }
    false
}

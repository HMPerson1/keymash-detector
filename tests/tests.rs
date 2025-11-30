use std::{fs, io, sync::Once};

use keymash_detector::*;

static LOGGER_INIT: Once = Once::new();

fn init_logger() {
    LOGGER_INIT.call_once(|| {
        env_logger::builder()
            .filter_level(log::LevelFilter::Warn)
            .init();
    });
}

#[test]
fn test_keymashes() -> io::Result<()> {
    init_logger();

    for (i, input_str) in fs::read_to_string("test-data/keymashes.txt")?
        .lines()
        .enumerate()
    {
        let input = preprocess(input_str.as_bytes());
        let res = fit_keymash_model(&input) - eval_english_model(&input);
        log::warn!("{}\n  {}", input_str, res);
        if ![26, 64, 65].contains(&i) {
            assert!(res > 2., "{:?}\n  {}", input_str, res);
        }
    }

    Ok(())
}

#[test]
fn test_chat_msgs() -> io::Result<()> {
    init_logger();

    for (_i, input_str) in fs::read_to_string("test-data/chat msgs.txt")?
        .lines()
        .enumerate()
    {
        let input = preprocess(input_str.as_bytes());
        let res = fit_keymash_model(&input) - eval_english_model(&input);
        log::warn!("{}\n  {}", input_str, res);
        if input.len() >= 6 {
            assert!(res < 1., "{:?}\n  {}", input_str, res);
        }
    }

    Ok(())
}

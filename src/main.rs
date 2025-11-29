use keymash_detector::*;

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Trace).init();
    dbg!(fit_keymash_model(b"asdfjkl;"));
    dbg!(fit_keymash_model(b"12340987"));
    dbg!(fit_keymash_model(b"ghionvr;jesdgvsrdnuojk;il"));
    dbg!(fit_keymash_model(b"vcoeimwanrvaeciowmnkr"));
}

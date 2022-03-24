use app::AppState;
use iced::{window, Application, Settings};

pub mod algorithms_async;
pub mod app;

fn main() -> iced::Result {
    let settings = Settings {
        window: window::Settings {
            size: (1100, 720),
            ..Default::default()
        },
        default_font: Some(include_bytes!("../assets/NotoSans-Regular.ttf")),
        default_text_size: 18,
        ..Default::default()
    };

    AppState::run(settings)
}

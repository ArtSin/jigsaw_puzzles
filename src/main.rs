use app::AppState;
use iced::{window, Application, Settings};

pub mod app;
pub mod genetic_algorithm;
pub mod genetic_algorithm_async;
pub mod image_processing;

fn main() -> iced::Result {
    let settings = Settings {
        window: window::Settings {
            size: (800, 600),
            ..Default::default()
        },
        default_font: Some(include_bytes!("../assets/NotoSans-Regular.ttf")),
        default_text_size: 18,
        ..Default::default()
    };

    AppState::run(settings)
}

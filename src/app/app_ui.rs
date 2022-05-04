use iced::{
    alignment, button, executor, scrollable, text_input, Alignment, Application, Button, Checkbox,
    Column, Command, Container, Element, Image, Length, Radio, Row, Scrollable, Space, Text,
    TextInput,
};
use iced_aw::{modal, Card, Modal};

use crate::algorithms_async::{Algorithm, AlgorithmData, AlgorithmState, CompatibilityMeasure};

use super::{
    images_loader::LoadImagesState, AppMessage, AppState, ErrorModalMessage, ErrorModalState,
};

#[derive(Default)]
pub struct AppUIState {
    pub error_modal_text: String,
    pub error_modal_state: modal::State<ErrorModalState>,

    open_file_button: button::State,
    piece_size_number_input: text_input::State,
    generations_count_number_input: text_input::State,
    population_size_number_input: text_input::State,
    rand_seed_number_input: text_input::State,
    start_algorithm_button: button::State,
    save_results_button: button::State,
    save_image_button: button::State,

    pub main_image_selected_image: Option<usize>,
    pub main_image_selected_generation: Option<usize>,
    pub main_image_handle: Option<iced::image::Handle>,
    pub main_image_direct_comparison: f32,
    pub main_image_neighbour_comparison: f32,
    pub show_incorrect_pieces: bool,
    pub show_incorrect_direct_neighbour: bool,
    first_generation_button: button::State,
    prev_generation_button: button::State,
    next_generation_button: button::State,
    last_generation_button: button::State,

    pub images_buttons: Vec<button::State>,
    images_scrollable: scrollable::State,
}

impl AppUIState {
    pub fn reset_state(&mut self, reset_images: bool) {
        self.main_image_selected_image = None;
        self.main_image_selected_generation = None;
        self.main_image_handle = None;

        if reset_images {
            self.images_buttons.clear();
            self.images_scrollable = scrollable::State::new();
        }
    }
}

impl Application for AppState {
    type Executor = executor::Default;
    type Message = AppMessage;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<AppMessage>) {
        (Self::default(), Command::none())
    }

    fn title(&self) -> String {
        String::from("Пазлы")
    }

    fn update(&mut self, message: AppMessage) -> Command<AppMessage> {
        match self.update_with_result(message) {
            Ok(command) => command,
            Err(error) => {
                let error_text = error.to_string();
                Command::perform(async {}, move |_| {
                    AppMessage::ErrorModalMessage(ErrorModalMessage::OpenModal(error_text.clone()))
                })
            }
        }
    }

    fn view(&mut self) -> Element<AppMessage> {
        let status_text = format!(
            "Статус: {}",
            match &self.algorithm_state {
                AlgorithmState::NotStarted => match &self.load_images_state {
                    LoadImagesState::NotLoaded => String::from("изображения не загружены"),
                    LoadImagesState::Preparing => String::from("подготовка..."),
                    LoadImagesState::Loading(data) => {
                        format!(
                            "загрузка изображения {}/{}",
                            data.loaded + 1,
                            data.paths.len()
                        )
                    }
                    LoadImagesState::Loaded(_) => String::from("изображения загружены"),
                },
                AlgorithmState::Running(algorithm_data) => match algorithm_data {
                    AlgorithmData::Genetic(algorithm_data) => match &self.load_images_state {
                        LoadImagesState::Loaded(images_data) => format!(
                            "обработка изображения {}/{}, поколение {}/{}",
                            algorithm_data.images_processed + 1,
                            images_data.images.len(),
                            algorithm_data.image_generations_processed + 1,
                            self.generations_count
                        ),
                        _ => unreachable!(),
                    },
                    AlgorithmData::LoopConstraints(algorithm_data) => match &self.load_images_state
                    {
                        LoadImagesState::Loaded(images_data) => format!(
                            "обработка изображения {}/{}",
                            algorithm_data.images_processed + 1,
                            images_data.images.len(),
                        ),
                        _ => unreachable!(),
                    },
                },
                AlgorithmState::Finished(_) => String::from("алгоритм выполнен"),
            }
        );

        // Меню (слева)
        let menu_column = Column::new()
            .width(Length::FillPortion(1))
            .padding(5)
            .spacing(10)
            .push({
                let button = Button::new(
                    &mut self.ui.open_file_button,
                    Container::new(Text::new("Открыть папку"))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x()
                        .center_y(),
                )
                .width(Length::Fill);
                match (&self.load_images_state, &self.algorithm_state) {
                    (
                        LoadImagesState::NotLoaded | LoadImagesState::Loaded(_),
                        AlgorithmState::NotStarted | AlgorithmState::Finished(_),
                    ) => button.on_press(AppMessage::LoadImagesPressed),
                    _ => button,
                }
            })
            .push(Text::new("Алгоритм:"))
            .push(Radio::new(
                Algorithm::Genetic,
                "Генетический",
                Some(self.algorithm),
                AppMessage::AlgorithmToggled,
            ))
            .push(Radio::new(
                Algorithm::LoopConstraints,
                "Циклических ограничений",
                Some(self.algorithm),
                AppMessage::AlgorithmToggled,
            ))
            .push(Text::new("Метод сравнения деталей:"))
            .push(Radio::new(
                CompatibilityMeasure::LabSSD,
                "LAB SSD",
                Some(self.compatibility_measure),
                AppMessage::CompatibilityMeasureToggled,
            ))
            .push(Radio::new(
                CompatibilityMeasure::MGC,
                "MGC",
                Some(self.compatibility_measure),
                AppMessage::CompatibilityMeasureToggled,
            ))
            .push(Text::new("Размер детали:"))
            .push(TextInput::new(
                &mut self.ui.piece_size_number_input,
                "",
                &self.piece_size.to_string(),
                AppMessage::PieceSizeChanged,
            ));
        let menu_column = match self.algorithm {
            Algorithm::Genetic => menu_column
                .push(Text::new("Количество поколений:"))
                .push(TextInput::new(
                    &mut self.ui.generations_count_number_input,
                    "",
                    &self.generations_count.to_string(),
                    AppMessage::GenerationsCountChanged,
                ))
                .push(Text::new("Размер популяции:"))
                .push(TextInput::new(
                    &mut self.ui.population_size_number_input,
                    "",
                    &self.population_size.to_string(),
                    AppMessage::PopulationSizeChanged,
                ))
                .push(Text::new("Начальное значение ГПСЧ:"))
                .push(TextInput::new(
                    &mut self.ui.rand_seed_number_input,
                    "",
                    &self.rand_seed.to_string(),
                    AppMessage::RandSeedChanged,
                )),
            Algorithm::LoopConstraints => menu_column,
        };
        let menu_column = menu_column
            .push({
                let button = Button::new(
                    &mut self.ui.start_algorithm_button,
                    Container::new(Text::new("Запустить алгоритм"))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x()
                        .center_y(),
                )
                .width(Length::Fill);
                match (&self.load_images_state, &self.algorithm_state) {
                    (
                        LoadImagesState::Loaded(_),
                        AlgorithmState::NotStarted | AlgorithmState::Finished(_),
                    ) => button.on_press(AppMessage::StartAlgorithmPressed),
                    _ => button,
                }
            })
            .push({
                let button = Button::new(
                    &mut self.ui.save_results_button,
                    Container::new(Text::new("Сохранить результаты"))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x()
                        .center_y(),
                )
                .width(Length::Fill);
                match (&self.load_images_state, &self.algorithm_state) {
                    (LoadImagesState::Loaded(_), AlgorithmState::Finished(_)) => {
                        button.on_press(AppMessage::SaveResultsPressed)
                    }
                    _ => button,
                }
            })
            .push({
                let button = Button::new(
                    &mut self.ui.save_image_button,
                    Container::new(Text::new("Сохранить изображение"))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x()
                        .center_y(),
                )
                .width(Length::Fill);
                if self.ui.main_image_selected_image.is_some() {
                    button.on_press(AppMessage::SaveImagePressed)
                } else {
                    button
                }
            })
            .push(Text::new(status_text));

        // Изображение поколения и информация о нём (центр)
        let mut main_image_column = Column::new()
            .width(Length::FillPortion(3))
            .padding(5)
            .spacing(10);

        if self.ui.main_image_selected_image.is_some() {
            let gen_cnt = match &self.algorithm_state {
                AlgorithmState::Finished(algorithm_data) => match algorithm_data {
                    AlgorithmData::Genetic(algorithm_data) => algorithm_data.best_chromosomes
                        [self.ui.main_image_selected_image.unwrap()]
                    .len(),
                    AlgorithmData::LoopConstraints(algorithm_data) => {
                        algorithm_data.solutions[self.ui.main_image_selected_image.unwrap()].len()
                    }
                },
                _ => unreachable!(),
            };

            main_image_column = main_image_column
                .push(
                    Container::new(
                        Image::new(self.ui.main_image_handle.clone().unwrap())
                            .width(Length::Fill)
                            .height(Length::Fill),
                    )
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x()
                    .center_y(),
                )
                .push(
                    Container::new(
                        Row::new()
                            .spacing(10)
                            .align_items(Alignment::Center)
                            .push({
                                let button = Button::new(
                                    &mut self.ui.first_generation_button,
                                    Container::new(Text::new("<<"))
                                        .width(Length::Fill)
                                        .height(Length::Fill)
                                        .center_x()
                                        .center_y(),
                                );
                                if self.ui.main_image_selected_generation.unwrap() > 0 {
                                    button.on_press(AppMessage::FirstGenerationPressed)
                                } else {
                                    button
                                }
                            })
                            .push({
                                let button = Button::new(
                                    &mut self.ui.prev_generation_button,
                                    Container::new(Text::new("<"))
                                        .width(Length::Fill)
                                        .height(Length::Fill)
                                        .center_x()
                                        .center_y(),
                                );
                                if self.ui.main_image_selected_generation.unwrap() > 0 {
                                    button.on_press(AppMessage::PrevGenerationPressed)
                                } else {
                                    button
                                }
                            })
                            .push(
                                Text::new(format!(
                                    "Поколение\n{} / {}",
                                    self.ui.main_image_selected_generation.unwrap() + 1,
                                    gen_cnt
                                ))
                                .horizontal_alignment(alignment::Horizontal::Center),
                            )
                            .push({
                                let button = Button::new(
                                    &mut self.ui.next_generation_button,
                                    Container::new(Text::new(">"))
                                        .width(Length::Fill)
                                        .height(Length::Fill)
                                        .center_x()
                                        .center_y(),
                                );
                                if self.ui.main_image_selected_generation.unwrap() + 1 < gen_cnt {
                                    button.on_press(AppMessage::NextGenerationPressed)
                                } else {
                                    button
                                }
                            })
                            .push({
                                let button = Button::new(
                                    &mut self.ui.last_generation_button,
                                    Container::new(Text::new(">>"))
                                        .width(Length::Fill)
                                        .height(Length::Fill)
                                        .center_x()
                                        .center_y(),
                                );
                                if self.ui.main_image_selected_generation.unwrap() + 1 < gen_cnt {
                                    button.on_press(AppMessage::LastGenerationPressed)
                                } else {
                                    button
                                }
                            }),
                    )
                    .width(Length::Fill)
                    .center_x(),
                )
                .push(
                    Container::new(
                        Row::new()
                            .spacing(10)
                            .align_items(Alignment::Center)
                            .push(Checkbox::new(
                                self.ui.show_incorrect_pieces,
                                "Показывать неправильные",
                                AppMessage::ShowIncorrectPiecesCheckboxToggled,
                            ))
                            .push(Radio::new(
                                false,
                                "Прямое сравнение",
                                Some(self.ui.show_incorrect_direct_neighbour),
                                AppMessage::ShowIncorrectDirectNeighbourToggled,
                            ))
                            .push(Radio::new(
                                true,
                                "Сравнение по соседям",
                                Some(self.ui.show_incorrect_direct_neighbour),
                                AppMessage::ShowIncorrectDirectNeighbourToggled,
                            )),
                    )
                    .width(Length::Fill)
                    .center_x(),
                )
                .push(
                    Container::new(Row::new().spacing(10).align_items(Alignment::Center).push(
                        Text::new(format!(
                            "Прямое сравнение: {:.2}%, сравнение по соседям: {:.2}%",
                            self.ui.main_image_direct_comparison,
                            self.ui.main_image_neighbour_comparison
                        )),
                    ))
                    .width(Length::Fill)
                    .center_x(),
                );
        } else {
            main_image_column = main_image_column.push(Space::new(Length::Fill, Length::Fill));
        }

        let mut images_scrollable = Scrollable::new(&mut self.ui.images_scrollable)
            .width(Length::FillPortion(1))
            .height(Length::Fill)
            .padding(5)
            .spacing(10);

        if let LoadImagesState::Loaded(data) = &self.load_images_state {
            for ((i, image_name), (image_handle, image_button)) in
                data.images_names.iter().enumerate().zip(
                    data.images_handles
                        .iter()
                        .zip(self.ui.images_buttons.iter_mut()),
                )
            {
                images_scrollable = images_scrollable.push({
                    let button = Button::new(
                        image_button,
                        Column::new()
                            .push(Image::new(image_handle.clone()).width(Length::Fill))
                            .push(
                                Container::new(Text::new(image_name))
                                    .width(Length::Fill)
                                    .center_x(),
                            ),
                    );
                    match self.algorithm_state {
                        AlgorithmState::Finished(_) => {
                            button.on_press(AppMessage::ImagesButtonPressed(i))
                        }
                        _ => button,
                    }
                });
            }
        }

        let content = Row::new()
            .padding(5)
            .spacing(10)
            .push(menu_column)
            .push(main_image_column)
            .push(images_scrollable);

        let content_container = Container::new(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x();

        Modal::new(&mut self.ui.error_modal_state, content_container, |state| {
            Card::new(Text::new("Ошибка"), Text::new(&self.ui.error_modal_text))
                .foot(
                    Row::new().spacing(10).padding(5).width(Length::Fill).push(
                        Button::new(
                            &mut state.ok_state,
                            Text::new("Ок").horizontal_alignment(alignment::Horizontal::Center),
                        )
                        .width(Length::Fill)
                        .on_press(AppMessage::ErrorModalMessage(
                            ErrorModalMessage::OkButtonPressed,
                        )),
                    ),
                )
                .max_width(300)
                .on_close(AppMessage::ErrorModalMessage(ErrorModalMessage::CloseModal))
                .into()
        })
        .backdrop(AppMessage::ErrorModalMessage(ErrorModalMessage::CloseModal))
        .on_esc(AppMessage::ErrorModalMessage(ErrorModalMessage::CloseModal))
        .into()
    }
}

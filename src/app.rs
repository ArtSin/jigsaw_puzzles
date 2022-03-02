use std::{error::Error, mem::swap, sync::Arc, time::Instant};

use iced::{button, Command};
use native_dialog::FileDialog;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use self::{
    app_ui::AppUIState,
    images_loader::{
        load_images_next, LoadImagesData, LoadImagesMessage, LoadImagesRequest, LoadImagesResponse,
        LoadImagesState,
    },
};
use crate::{
    genetic_algorithm_async::{
        algorithm_next, AlgorithmData, AlgorithmDataRequest, AlgorithmError, AlgorithmMessage,
        AlgorithmState,
    },
    image_processing::{
        get_chromosome_image, get_image_handle, image_direct_comparison, image_neighbour_comparison,
    },
};

mod app_ui;
mod images_loader;

pub struct AppState {
    piece_size: u32,
    generations_count: usize,
    population_size: usize,
    rand_seed: u64,

    load_images_state: LoadImagesState,
    algorithm_state: AlgorithmState,

    algorithm_start_time: Option<Instant>,

    ui: AppUIState,
}

#[derive(Debug, Clone)]
pub enum AppMessage {
    LoadImagesPressed,
    StartAlgorithmPressed,
    FirstGenerationPressed,
    PrevGenerationPressed,
    NextGenerationPressed,
    LastGenerationPressed,
    ImagesButtonPressed(usize),

    PieceSizeChanged(u32),
    GenerationsCountChanged(usize),
    PopulationSizeChanged(usize),
    RandSeedChanged(u64),

    LoadImagesMessage(LoadImagesMessage),
    AlgorithmMessage(AlgorithmMessage),
    ErrorModalMessage(ErrorModalMessage),
}

#[derive(Default)]
pub struct ErrorModalState {
    pub ok_state: button::State,
}

#[derive(Debug, Clone)]
pub enum ErrorModalMessage {
    OpenModal(String),
    CloseModal,
    OkButtonPressed,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            piece_size: 28,
            generations_count: 100,
            population_size: 100,
            rand_seed: 1,
            load_images_state: LoadImagesState::NotLoaded,
            algorithm_state: AlgorithmState::NotStarted,
            algorithm_start_time: None,
            ui: Default::default(),
        }
    }
}

impl AppState {
    pub fn load_images_start(&self) -> Result<Command<AppMessage>, Box<dyn Error>> {
        let dir_path_option = FileDialog::new().show_open_single_dir()?;
        if dir_path_option.is_none() {
            return Ok(Command::none());
        }

        Ok(Command::perform(async {}, move |_| {
            AppMessage::LoadImagesMessage(LoadImagesMessage::Preparation(
                dir_path_option.as_ref().unwrap().clone(),
            ))
        }))
    }

    pub fn algorithm_start(&self) -> Result<Command<AppMessage>, Box<dyn Error>> {
        let images_data = match &self.load_images_state {
            LoadImagesState::Loaded(images_data) => images_data,
            _ => unreachable!(),
        };

        let (image_width, image_height) = images_data
            .images
            .first()
            .ok_or(AlgorithmError::NoImages)?
            .dimensions();
        if images_data
            .images
            .iter()
            .any(|image| image.width() != image_width || image.height() != image_height)
        {
            return Err(Box::new(AlgorithmError::NotEqualDimensions));
        }
        if self.piece_size < 10
            || image_width % self.piece_size != 0
            || image_height % self.piece_size != 0
        {
            return Err(Box::new(AlgorithmError::IncorrectPieceSize));
        }
        let (img_width, img_height) = (
            (image_width / self.piece_size) as usize,
            (image_height / self.piece_size) as usize,
        );

        let algorithm_data = AlgorithmDataRequest {
            piece_size: self.piece_size,
            generations_count: self.generations_count,
            population_size: self.population_size,
            rng: Xoshiro256PlusPlus::seed_from_u64(self.rand_seed),
            img_width,
            img_height,
            images_processed: 0,
            image_generations_processed: 0,
            image_prepared: false,
            pieces_dissimilarity: Arc::new([Vec::new(), Vec::new()]),
            pieces_buddies: Arc::new([Vec::new(), Vec::new(), Vec::new(), Vec::new()]),
            current_generation: Arc::new(Vec::new()),
        };

        Ok(Command::perform(async {}, move |_| {
            AppMessage::AlgorithmMessage(AlgorithmMessage::Initialization(algorithm_data.clone()))
        }))
    }

    fn load_selected_image(&mut self) -> Result<Command<AppMessage>, Box<dyn Error>> {
        let images_data = match &self.load_images_state {
            LoadImagesState::Loaded(images_data) => images_data,
            _ => unreachable!(),
        };

        let image_i = self.ui.main_image_selected_image.unwrap();
        let gen = self.ui.main_image_selected_generation.unwrap();
        match &self.algorithm_state {
            AlgorithmState::Finished(algorithm_data) => {
                self.ui.main_image_selected_image = Some(image_i);
                self.ui.main_image_selected_generation = Some(gen);
                let new_image = get_chromosome_image(
                    &images_data.images[image_i],
                    self.piece_size,
                    &algorithm_data.best_chromosomes[image_i][gen],
                );
                self.ui.main_image_handle = Some(get_image_handle(&new_image));
                self.ui.main_image_direct_comparison =
                    image_direct_comparison(&algorithm_data.best_chromosomes[image_i][gen]);
                self.ui.main_image_neighbour_comparison =
                    image_neighbour_comparison(&algorithm_data.best_chromosomes[image_i][gen]);
                Ok(Command::none())
            }
            _ => unreachable!(),
        }
    }

    fn update_with_result(
        &mut self,
        message: AppMessage,
    ) -> Result<Command<AppMessage>, Box<dyn Error>> {
        match message {
            AppMessage::LoadImagesPressed => self.load_images_start(),
            AppMessage::StartAlgorithmPressed => self.algorithm_start(),
            AppMessage::FirstGenerationPressed => {
                self.ui.main_image_selected_generation = Some(0);
                self.load_selected_image()
            }
            AppMessage::PrevGenerationPressed => {
                self.ui.main_image_selected_generation =
                    Some(self.ui.main_image_selected_generation.unwrap() - 1);
                self.load_selected_image()
            }
            AppMessage::NextGenerationPressed => {
                self.ui.main_image_selected_generation =
                    Some(self.ui.main_image_selected_generation.unwrap() + 1);
                self.load_selected_image()
            }
            AppMessage::LastGenerationPressed => {
                self.ui.main_image_selected_generation = Some(self.generations_count - 1);
                self.load_selected_image()
            }
            AppMessage::ImagesButtonPressed(image_i) => {
                self.ui.main_image_selected_image = Some(image_i);
                self.ui.main_image_selected_generation = Some(0);
                self.load_selected_image()
            }

            AppMessage::PieceSizeChanged(num) => {
                self.piece_size = num;
                Ok(Command::none())
            }
            AppMessage::GenerationsCountChanged(num) => {
                self.generations_count = num;
                Ok(Command::none())
            }
            AppMessage::PopulationSizeChanged(num) => {
                self.population_size = num;
                Ok(Command::none())
            }
            AppMessage::RandSeedChanged(seed) => {
                self.rand_seed = seed;
                Ok(Command::none())
            }

            AppMessage::LoadImagesMessage(load_images_message) => match load_images_message {
                LoadImagesMessage::Preparation(dir_path) => {
                    self.load_images_state = LoadImagesState::Preparing;
                    self.algorithm_state = AlgorithmState::NotStarted;
                    self.ui.reset_state(true);
                    load_images_next(LoadImagesRequest::Prepare(dir_path))
                }
                LoadImagesMessage::Update(response) => match response {
                    LoadImagesResponse::Prepared(paths) => {
                        if paths.is_empty() {
                            self.load_images_state =
                                LoadImagesState::Loaded(LoadImagesData::create_from_paths(paths));
                            Ok(Command::none())
                        } else {
                            let first_path = paths.first().unwrap().clone();
                            self.load_images_state =
                                LoadImagesState::Loading(LoadImagesData::create_from_paths(paths));
                            load_images_next(LoadImagesRequest::Load(first_path))
                        }
                    }
                    LoadImagesResponse::Loaded(response_data) => {
                        let is_loaded = match &mut self.load_images_state {
                            LoadImagesState::Loading(data) => {
                                data.update_with_response(response_data);
                                data.loaded == data.paths.len()
                            }
                            _ => unreachable!(),
                        };
                        if is_loaded {
                            let mut state = LoadImagesState::NotLoaded;
                            swap(&mut state, &mut self.load_images_state);
                            state = LoadImagesState::Loaded(match state {
                                LoadImagesState::Loading(data) => {
                                    self.ui.images_buttons =
                                        vec![button::State::default(); data.paths.len()];
                                    data
                                }
                                _ => unreachable!(),
                            });
                            swap(&mut state, &mut self.load_images_state);
                            Ok(Command::none())
                        } else {
                            let curr_path = match &self.load_images_state {
                                LoadImagesState::Loading(data) => data.paths[data.loaded].clone(),
                                _ => unreachable!(),
                            };
                            load_images_next(LoadImagesRequest::Load(curr_path))
                        }
                    }
                },
                LoadImagesMessage::Error(error) => {
                    self.load_images_state = LoadImagesState::NotLoaded;
                    self.algorithm_state = AlgorithmState::NotStarted;
                    self.ui.reset_state(true);
                    Err(error.into())
                }
            },
            AppMessage::AlgorithmMessage(algorithm_message) => {
                let images_data = match &self.load_images_state {
                    LoadImagesState::Loaded(images_data) => images_data,
                    _ => unreachable!(),
                };
                match algorithm_message {
                    AlgorithmMessage::Initialization(request) => {
                        self.algorithm_start_time = Some(Instant::now());

                        self.algorithm_state = AlgorithmState::Running(
                            AlgorithmData::create_from_request(request.clone()),
                        );
                        self.ui.reset_state(false);
                        algorithm_next(Arc::clone(&images_data.images), request)
                    }
                    AlgorithmMessage::Update(response) => match &mut self.algorithm_state {
                        AlgorithmState::Running(data) => {
                            data.update_with_response(response);
                            algorithm_next(Arc::clone(&images_data.images), data.create_request())
                        }
                        _ => unreachable!(),
                    },
                    AlgorithmMessage::Finished(response) => {
                        let mut state = AlgorithmState::NotStarted;
                        swap(&mut state, &mut self.algorithm_state);
                        state = AlgorithmState::Finished(match state {
                            AlgorithmState::Running(mut data) => {
                                data.update_with_response(response);
                                data
                            }
                            _ => unreachable!(),
                        });
                        swap(&mut state, &mut self.algorithm_state);

                        let algorithm_duration =
                            Instant::now() - self.algorithm_start_time.unwrap();
                        println!(
                            "Время выполнения алгоритма: {:.6} с",
                            algorithm_duration.as_secs_f32()
                        );
                        Ok(Command::none())
                    }
                    AlgorithmMessage::Error(error) => {
                        self.algorithm_state = AlgorithmState::NotStarted;
                        self.ui.reset_state(false);
                        Err(error.into())
                    }
                }
            }
            AppMessage::ErrorModalMessage(error_modal_message) => match error_modal_message {
                ErrorModalMessage::OpenModal(text) => {
                    self.ui.error_modal_text = text;
                    self.ui.error_modal_state.show(true);
                    Ok(Command::none())
                }
                ErrorModalMessage::CloseModal | ErrorModalMessage::OkButtonPressed => {
                    self.ui.error_modal_state.show(false);
                    Ok(Command::none())
                }
            },
        }
    }
}

use std::{error::Error, fmt::Display, sync::Arc};

use iced::Command;
use image::RgbaImage;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{
    app::AppMessage,
    genetic_algorithm::{algorithm_step, calculate_dissimilarities, find_best_buddies, Chromosome},
};

#[derive(Debug, Clone)]
pub struct AlgorithmDataRequest {
    pub piece_size: u32,
    pub generations_count: usize,
    pub population_size: usize,
    pub rng: Xoshiro256PlusPlus,
    pub img_width: usize,
    pub img_height: usize,
    pub images_processed: usize,
    pub image_generations_processed: usize,
    pub image_prepared: bool,
    pub pieces_dissimilarity: Arc<[Vec<Vec<f32>>; 2]>,
    pub pieces_buddies: Arc<[Vec<(usize, usize)>; 4]>,
    pub current_generation: Arc<Vec<Chromosome>>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmDataResponse {
    pub rng: Xoshiro256PlusPlus,
    pub images_processed: usize,
    pub image_generations_processed: usize,
    pub pieces_dissimilarity: Option<[Vec<Vec<f32>>; 2]>,
    pub pieces_buddies: Option<[Vec<(usize, usize)>; 4]>,
    pub current_generation: Option<Vec<Chromosome>>,
    pub best_chromosome: Option<Chromosome>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmData {
    pub piece_size: u32,
    pub generations_count: usize,
    pub population_size: usize,
    pub rng: Xoshiro256PlusPlus,
    pub img_width: usize,
    pub img_height: usize,
    pub images_processed: usize,
    pub image_generations_processed: usize,
    pub pieces_dissimilarity: Arc<[Vec<Vec<f32>>; 2]>,
    pub pieces_buddies: Arc<[Vec<(usize, usize)>; 4]>,
    pub current_generation: Arc<Vec<Chromosome>>,
    pub best_chromosomes: Vec<Vec<Chromosome>>, // лучшая хромосома для каждого изображения и поколения
}

impl AlgorithmData {
    pub fn create_request(&self) -> AlgorithmDataRequest {
        AlgorithmDataRequest {
            piece_size: self.piece_size,
            generations_count: self.generations_count,
            population_size: self.population_size,
            rng: self.rng.clone(),
            img_width: self.img_width,
            img_height: self.img_height,
            images_processed: self.images_processed,
            image_generations_processed: self.image_generations_processed,
            image_prepared: !self.pieces_dissimilarity[0].is_empty(),
            pieces_dissimilarity: Arc::clone(&self.pieces_dissimilarity),
            pieces_buddies: Arc::clone(&self.pieces_buddies),
            current_generation: Arc::clone(&self.current_generation),
        }
    }

    pub fn create_from_request(request: AlgorithmDataRequest) -> Self {
        Self {
            piece_size: request.piece_size,
            generations_count: request.generations_count,
            population_size: request.population_size,
            rng: request.rng,
            img_width: request.img_width,
            img_height: request.img_height,
            images_processed: request.images_processed,
            image_generations_processed: request.image_generations_processed,
            pieces_dissimilarity: request.pieces_dissimilarity,
            pieces_buddies: request.pieces_buddies,
            current_generation: request.current_generation,
            best_chromosomes: Vec::new(),
        }
    }

    pub fn update_with_response(&mut self, response: AlgorithmDataResponse) {
        self.rng = response.rng;
        self.images_processed = response.images_processed;
        self.image_generations_processed = response.image_generations_processed;
        if let Some(pieces_compatibility) = response.pieces_dissimilarity {
            self.pieces_dissimilarity = Arc::new(pieces_compatibility);
        }
        if let Some(pieces_buddies) = response.pieces_buddies {
            self.pieces_buddies = Arc::new(pieces_buddies);
        }
        if let Some(generation) = response.current_generation {
            self.current_generation = Arc::new(generation);
        }
        if let Some(chromosome) = response.best_chromosome {
            if self.images_processed == self.best_chromosomes.len() {
                self.best_chromosomes.push(Vec::new());
            }
            self.best_chromosomes.last_mut().unwrap().push(chromosome);
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlgorithmState {
    NotStarted,
    Running(AlgorithmData),
    Finished(AlgorithmData),
}

#[derive(Debug, Clone)]
pub enum AlgorithmMessage {
    Initialization(AlgorithmDataRequest),
    Update(AlgorithmDataResponse),
    Finished(AlgorithmDataResponse),
    Error(String),
}

#[derive(Debug)]
pub enum AlgorithmError {
    NoImages,
    NotEqualDimensions,
    IncorrectPieceSize,
}

impl Display for AlgorithmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoImages => write!(f, "Нет изображений!"),
            Self::NotEqualDimensions => write!(f, "У изображений не одинаковые размеры!"),
            Self::IncorrectPieceSize => write!(f, "Неправильный размер детали!"),
        }
    }
}

impl Error for AlgorithmError {}

pub fn algorithm_next(
    images: Arc<Vec<RgbaImage>>,
    mut request: AlgorithmDataRequest,
) -> Result<Command<AppMessage>, Box<dyn Error>> {
    let future = async move {
        let res = || {
            // Все изображения обработаны
            if request.images_processed == images.len() {
                return Ok(AlgorithmMessage::Finished(AlgorithmDataResponse {
                    rng: request.rng,
                    images_processed: request.images_processed,
                    image_generations_processed: request.image_generations_processed,
                    pieces_dissimilarity: None,
                    pieces_buddies: None,
                    current_generation: None,
                    best_chromosome: None,
                }));
            }
            // Все поколения изображения обработаны
            if request.image_generations_processed == request.generations_count {
                return Ok(AlgorithmMessage::Update(AlgorithmDataResponse {
                    rng: request.rng,
                    images_processed: request.images_processed + 1,
                    image_generations_processed: 0,
                    pieces_dissimilarity: Some([Vec::new(), Vec::new()]),
                    pieces_buddies: Some([Vec::new(), Vec::new(), Vec::new(), Vec::new()]),
                    current_generation: None,
                    best_chromosome: None,
                }));
            }
            // Подготовка - вычисление совместимостей деталей
            if !request.image_prepared {
                let pieces_dissimilarity = calculate_dissimilarities(
                    &images[request.images_processed],
                    request.img_width,
                    request.img_height,
                    request.piece_size as usize,
                );
                let pieces_buddies = find_best_buddies(request.img_width, &pieces_dissimilarity);
                return Ok(AlgorithmMessage::Update(AlgorithmDataResponse {
                    rng: request.rng,
                    images_processed: request.images_processed,
                    image_generations_processed: request.image_generations_processed,
                    pieces_dissimilarity: Some(pieces_dissimilarity),
                    pieces_buddies: Some(pieces_buddies),
                    current_generation: None,
                    best_chromosome: None,
                }));
            }

            let new_generation = algorithm_step(
                request.population_size,
                &mut request.rng,
                request.img_width,
                request.img_height,
                request.image_generations_processed,
                &request.pieces_dissimilarity,
                &request.pieces_buddies,
                &request.current_generation,
            );

            Ok::<_, Box<dyn Error>>(AlgorithmMessage::Update(AlgorithmDataResponse {
                rng: request.rng,
                images_processed: request.images_processed,
                image_generations_processed: request.image_generations_processed + 1,
                pieces_dissimilarity: None,
                pieces_buddies: None,
                best_chromosome: Some(new_generation[0].clone()),
                current_generation: Some(new_generation),
            }))
        };
        match res() {
            Ok(message) => AppMessage::AlgorithmMessage(message),
            Err(error) => AppMessage::AlgorithmMessage(AlgorithmMessage::Error(error.to_string())),
        }
    };
    Ok(Command::from(future))
}

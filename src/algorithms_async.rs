use std::{error::Error, fmt::Display, sync::Arc};

use iced::Command;
use image::RgbaImage;
use jigsaw_puzzles::{calculate_lab_ssd, calculate_mgc};

use crate::app::AppMessage;

pub mod genetic_algorithm;
pub mod loop_constraints_algorithm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Genetic,
    LoopConstraints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityMeasure {
    LabSSD,
    MGC,
}

impl CompatibilityMeasure {
    pub fn calculate(
        &self,
        image: &RgbaImage,
        img_width: usize,
        img_height: usize,
        piece_size: usize,
    ) -> [Vec<Vec<f32>>; 2] {
        match self {
            Self::LabSSD => calculate_lab_ssd(image, img_width, img_height, piece_size),
            Self::MGC => calculate_mgc(image, img_width, img_height, piece_size),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlgorithmDataRequest {
    Genetic(genetic_algorithm::AlgorithmDataRequest),
    LoopConstraints(loop_constraints_algorithm::AlgorithmDataRequest),
}

#[derive(Debug, Clone)]
pub enum AlgorithmDataResponse {
    Genetic(genetic_algorithm::AlgorithmDataResponse),
    LoopConstraints(loop_constraints_algorithm::AlgorithmDataResponse),
}

#[derive(Debug, Clone)]
pub enum AlgorithmData {
    Genetic(genetic_algorithm::AlgorithmData),
    LoopConstraints(loop_constraints_algorithm::AlgorithmData),
}

impl AlgorithmData {
    pub fn create_request(&self) -> AlgorithmDataRequest {
        match self {
            Self::Genetic(algorithm_data) => {
                AlgorithmDataRequest::Genetic(algorithm_data.create_request())
            }
            Self::LoopConstraints(algorithm_data) => {
                AlgorithmDataRequest::LoopConstraints(algorithm_data.create_request())
            }
        }
    }

    pub fn create_from_request(request: AlgorithmDataRequest) -> Self {
        match request {
            AlgorithmDataRequest::Genetic(request) => Self::Genetic(
                genetic_algorithm::AlgorithmData::create_from_request(request),
            ),
            AlgorithmDataRequest::LoopConstraints(request) => Self::LoopConstraints(
                loop_constraints_algorithm::AlgorithmData::create_from_request(request),
            ),
        }
    }

    pub fn update_with_response(&mut self, response: AlgorithmDataResponse) {
        match (self, response) {
            (Self::Genetic(algorithm_data), AlgorithmDataResponse::Genetic(response)) => {
                algorithm_data.update_with_response(response)
            }
            (
                Self::LoopConstraints(algorithm_data),
                AlgorithmDataResponse::LoopConstraints(response),
            ) => algorithm_data.update_with_response(response),
            _ => unreachable!(),
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
    request: AlgorithmDataRequest,
) -> Result<Command<AppMessage>, Box<dyn Error>> {
    match request {
        AlgorithmDataRequest::Genetic(request) => {
            genetic_algorithm::algorithm_next(images, request)
        }
        AlgorithmDataRequest::LoopConstraints(request) => {
            loop_constraints_algorithm::algorithm_next(images, request)
        }
    }
}

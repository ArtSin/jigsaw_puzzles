use std::{error::Error, sync::Arc};

use iced::Command;
use image::RgbaImage;
use jigsaw_puzzles::{
    loop_constraints_algorithm::{algorithm_step, find_match_candidates},
    Solution,
};

use crate::app::AppMessage;

use super::{AlgorithmMessage, CompatibilityMeasure};

#[derive(Debug, Clone)]
pub struct AlgorithmDataRequest {
    pub compatibility_measure: CompatibilityMeasure,
    pub piece_size: u32,
    pub img_width: usize,
    pub img_height: usize,
    pub images_processed: usize,
    pub image_prepared: bool,
    pub pieces_compatibility: Arc<[Vec<Vec<f32>>; 2]>,
    pub pieces_match_candidates: Arc<[Vec<Vec<(usize, usize)>>; 4]>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmDataResponse {
    pub images_processed: usize,
    pub pieces_compatibility: Option<[Vec<Vec<f32>>; 2]>,
    pub pieces_match_candidates: Option<[Vec<Vec<(usize, usize)>>; 4]>,
    pub solutions: Option<Vec<Solution>>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmData {
    pub compatibility_measure: CompatibilityMeasure,
    pub piece_size: u32,
    pub img_width: usize,
    pub img_height: usize,
    pub images_processed: usize,
    pub pieces_compatibility: Arc<[Vec<Vec<f32>>; 2]>,
    pub pieces_match_candidates: Arc<[Vec<Vec<(usize, usize)>>; 4]>,
    pub solutions: Vec<Vec<Solution>>,
}

impl AlgorithmData {
    pub fn create_request(&self) -> AlgorithmDataRequest {
        AlgorithmDataRequest {
            compatibility_measure: self.compatibility_measure,
            piece_size: self.piece_size,
            img_width: self.img_width,
            img_height: self.img_height,
            images_processed: self.images_processed,
            image_prepared: !self.pieces_compatibility[0].is_empty(),
            pieces_compatibility: Arc::clone(&self.pieces_compatibility),
            pieces_match_candidates: Arc::clone(&self.pieces_match_candidates),
        }
    }

    pub fn create_from_request(request: AlgorithmDataRequest) -> Self {
        Self {
            compatibility_measure: request.compatibility_measure,
            piece_size: request.piece_size,
            img_width: request.img_width,
            img_height: request.img_height,
            images_processed: request.images_processed,
            pieces_compatibility: request.pieces_compatibility,
            pieces_match_candidates: request.pieces_match_candidates,
            solutions: Vec::new(),
        }
    }

    pub fn update_with_response(&mut self, response: AlgorithmDataResponse) {
        self.images_processed = response.images_processed;
        if let Some(pieces_compatibility) = response.pieces_compatibility {
            self.pieces_compatibility = Arc::new(pieces_compatibility);
        }
        if let Some(pieces_match_candidates) = response.pieces_match_candidates {
            self.pieces_match_candidates = Arc::new(pieces_match_candidates);
        }
        if let Some(solutions) = response.solutions {
            self.solutions.push(solutions);
        }
    }
}

pub fn algorithm_next(
    images: Arc<Vec<RgbaImage>>,
    request: AlgorithmDataRequest,
) -> Result<Command<AppMessage>, Box<dyn Error>> {
    let future = async move {
        let res = || {
            // Все изображения обработаны
            if request.images_processed == images.len() {
                return Ok(AlgorithmMessage::Finished(
                    super::AlgorithmDataResponse::LoopConstraints(AlgorithmDataResponse {
                        images_processed: request.images_processed,
                        pieces_compatibility: None,
                        pieces_match_candidates: None,
                        solutions: None,
                    }),
                ));
            }
            // Подготовка - вычисление совместимостей деталей
            if !request.image_prepared {
                let pieces_compatibility = request.compatibility_measure.calculate(
                    &images[request.images_processed],
                    request.img_width,
                    request.img_height,
                    request.piece_size as usize,
                );
                let pieces_match_candidates =
                    find_match_candidates(request.img_width, &pieces_compatibility);
                return Ok(AlgorithmMessage::Update(
                    super::AlgorithmDataResponse::LoopConstraints(AlgorithmDataResponse {
                        images_processed: request.images_processed,
                        pieces_compatibility: Some(pieces_compatibility),
                        pieces_match_candidates: Some(pieces_match_candidates),
                        solutions: None,
                    }),
                ));
            }

            let new_solution = algorithm_step(
                request.img_width,
                request.img_height,
                &request.pieces_compatibility,
                &request.pieces_match_candidates,
            );

            Ok::<_, Box<dyn Error>>(AlgorithmMessage::Update(
                super::AlgorithmDataResponse::LoopConstraints(AlgorithmDataResponse {
                    images_processed: request.images_processed + 1,
                    pieces_compatibility: None,
                    pieces_match_candidates: None,
                    solutions: Some(new_solution),
                }),
            ))
        };
        match res() {
            Ok(message) => AppMessage::AlgorithmMessage(message),
            Err(error) => AppMessage::AlgorithmMessage(AlgorithmMessage::Error(error.to_string())),
        }
    };
    Ok(Command::from(future))
}

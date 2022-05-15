use std::{error::Error, sync::Arc};

use iced::Command;
use image::RgbaImage;
use jigsaw_puzzles::{
    loop_constraints_algorithm::{algorithm_step, find_match_candidates},
    Solution,
};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::app::AppMessage;

use super::{AlgorithmMessage, CompatibilityMeasure};

#[derive(Debug, Clone)]
pub struct AlgorithmDataRequest {
    pub compatibility_measure: CompatibilityMeasure,
    pub piece_size: u32,
    pub rng: Xoshiro256PlusPlus,
    pub img_width: usize,
    pub img_height: usize,
    pub images_processed: usize,
}

#[derive(Debug, Clone)]
pub struct AlgorithmDataResponse {
    pub images_processed: usize,
    pub solutions: Option<Vec<Solution>>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmData {
    pub compatibility_measure: CompatibilityMeasure,
    pub piece_size: u32,
    pub rng: Xoshiro256PlusPlus,
    pub img_width: usize,
    pub img_height: usize,
    pub images_processed: usize,
    pub solutions: Vec<Vec<Solution>>,
}

impl AlgorithmData {
    pub fn create_request(&self) -> AlgorithmDataRequest {
        AlgorithmDataRequest {
            compatibility_measure: self.compatibility_measure,
            piece_size: self.piece_size,
            rng: self.rng.clone(),
            img_width: self.img_width,
            img_height: self.img_height,
            images_processed: self.images_processed,
        }
    }

    pub fn create_from_request(request: AlgorithmDataRequest) -> Self {
        Self {
            compatibility_measure: request.compatibility_measure,
            piece_size: request.piece_size,
            rng: request.rng,
            img_width: request.img_width,
            img_height: request.img_height,
            images_processed: request.images_processed,
            solutions: Vec::new(),
        }
    }

    pub fn update_with_response(&mut self, response: AlgorithmDataResponse) {
        self.images_processed = response.images_processed;
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
        // Все изображения обработаны
        if request.images_processed == images.len() {
            return Ok(AlgorithmMessage::Finished(
                super::AlgorithmDataResponse::LoopConstraints(AlgorithmDataResponse {
                    images_processed: request.images_processed,
                    solutions: None,
                }),
            ));
        }
        // Подготовка - вычисление совместимостей деталей
        let pieces_compatibility = request.compatibility_measure.calculate(
            &images[request.images_processed],
            request.img_width,
            request.img_height,
            request.piece_size as usize,
        );
        let pieces_match_candidates =
            find_match_candidates(request.img_width, &pieces_compatibility);

        let new_solution = algorithm_step(
            request.img_width,
            request.img_height,
            &pieces_compatibility,
            &pieces_match_candidates,
        );

        Ok(AlgorithmMessage::Update(
            super::AlgorithmDataResponse::LoopConstraints(AlgorithmDataResponse {
                images_processed: request.images_processed + 1,
                solutions: Some(new_solution),
            }),
        ))
    };
    let f = |res: Result<AlgorithmMessage, Box<dyn Error>>| match res {
        Ok(message) => AppMessage::AlgorithmMessage(message),
        Err(error) => AppMessage::AlgorithmMessage(AlgorithmMessage::Error(error.to_string())),
    };
    Ok(Command::perform(future, f))
}

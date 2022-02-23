use iced::image::Handle;
use image::{GenericImage, GenericImageView, ImageBuffer, RgbaImage};
use lab::Lab;

use crate::genetic_algorithm::Chromosome;

pub fn get_image_handle(image: &RgbaImage) -> Handle {
    // RGBA -> BGRA
    Handle::from_pixels(
        image.width(),
        image.height(),
        image
            .as_raw()
            .chunks_exact(4)
            .flat_map(|arr| [arr[2], arr[1], arr[0], arr[3]])
            .collect(),
    )
}

pub fn get_chromosome_image(
    image: &RgbaImage,
    piece_size: u32,
    chromosome: &Chromosome,
) -> RgbaImage {
    let pieces: Vec<Vec<_>> = (0..(image.height() / piece_size))
        .map(|r| {
            (0..(image.width() / piece_size))
                .map(|c| image.view(c * piece_size, r * piece_size, piece_size, piece_size))
                .collect()
        })
        .collect();

    let mut new_image = ImageBuffer::new(image.width(), image.height());
    for (r, v) in chromosome.iter().enumerate() {
        for (c, (i, j)) in v.iter().enumerate() {
            new_image
                .copy_from(
                    &*pieces[*i][*j],
                    (c as u32) * piece_size,
                    (r as u32) * piece_size,
                )
                .unwrap();
        }
    }
    new_image
}

pub fn get_lab_image(image: &RgbaImage) -> Vec<Lab> {
    let rgb_pixels: Vec<_> = image
        .as_raw()
        .clone()
        .into_iter()
        .enumerate()
        .filter_map(|(i, x)| if i % 4 != 0 { Some(x) } else { None })
        .collect();
    lab::rgb_bytes_to_labs(&rgb_pixels)
}

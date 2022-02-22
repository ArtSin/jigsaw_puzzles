use std::{error::Error, fmt::Display, fs::read_dir, mem::swap, path::PathBuf, sync::Arc};

use iced::Command;
use image::RgbaImage;

use super::AppMessage;
use crate::image_processing::get_image_handle;

#[derive(Debug, Clone)]
pub enum LoadImagesRequest {
    Prepare(PathBuf),
    Load(PathBuf),
}

#[derive(Debug, Clone)]
pub struct LoadImagesResponseData {
    pub image: RgbaImage,
    pub image_name: String,
    pub image_handle: iced::image::Handle,
}

#[derive(Debug, Clone)]
pub enum LoadImagesResponse {
    Prepared(Vec<PathBuf>),
    Loaded(LoadImagesResponseData),
}

#[derive(Debug, Clone)]
pub struct LoadImagesData {
    pub paths: Vec<PathBuf>,
    pub images: Arc<Vec<RgbaImage>>,
    pub images_names: Vec<String>,
    pub images_handles: Vec<iced::image::Handle>,
    pub loaded: usize,
}

impl LoadImagesData {
    pub fn create_from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            images: Arc::new(Vec::new()),
            images_names: Vec::new(),
            images_handles: Vec::new(),
            loaded: 0,
        }
    }

    pub fn update_with_response(&mut self, response_data: LoadImagesResponseData) {
        let mut tmp = Arc::new(Vec::new());
        swap(&mut tmp, &mut self.images);
        let mut images = Arc::try_unwrap(tmp).unwrap();
        images.push(response_data.image);
        self.images = Arc::new(images);
        self.images_names.push(response_data.image_name);
        self.images_handles.push(response_data.image_handle);
        self.loaded += 1;
    }
}

#[derive(Debug, Clone)]
pub enum LoadImagesState {
    NotLoaded,
    Preparing,
    Loading(LoadImagesData),
    Loaded(LoadImagesData),
}

#[derive(Debug, Clone)]
pub enum LoadImagesMessage {
    Preparation(PathBuf),
    Update(LoadImagesResponse),
    Error(String),
}

#[derive(Debug)]
enum LoadImagesError {
    IncorrectFileName,
}

impl Display for LoadImagesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncorrectFileName => write!(f, "Неправильное имя файла!"),
        }
    }
}

impl Error for LoadImagesError {}

pub fn load_images_next(request: LoadImagesRequest) -> Result<Command<AppMessage>, Box<dyn Error>> {
    let future = async {
        let res = || match request {
            LoadImagesRequest::Prepare(dir_path) => {
                let mut paths: Vec<_> = read_dir(dir_path)?
                    .filter(|dir_entry_res| {
                        dir_entry_res.is_err() || dir_entry_res.as_ref().unwrap().path().is_file()
                    })
                    .map(|dir_entry_res| dir_entry_res.map(|dir_entry| dir_entry.path()))
                    .collect::<Result<_, _>>()?;
                alphanumeric_sort::sort_path_slice(&mut paths);

                Ok::<_, Box<dyn Error>>(LoadImagesMessage::Update(LoadImagesResponse::Prepared(
                    paths,
                )))
            }
            LoadImagesRequest::Load(path) => {
                let image = image::open(&path)?.to_rgba8();
                let image_name = path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .ok_or(LoadImagesError::IncorrectFileName)?
                    .to_string();
                let image_handle = get_image_handle(&image);
                Ok(LoadImagesMessage::Update(LoadImagesResponse::Loaded(
                    LoadImagesResponseData {
                        image,
                        image_name,
                        image_handle,
                    },
                )))
            }
        };
        match res() {
            Ok(message) => AppMessage::LoadImagesMessage(message),
            Err(error) => {
                AppMessage::LoadImagesMessage(LoadImagesMessage::Error(error.to_string()))
            }
        }
    };
    Ok(Command::from(future))
}

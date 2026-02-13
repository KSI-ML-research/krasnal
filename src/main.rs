use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use pgn_reader::{Visitor, SanPlus, Reader, RawTag}; 
use polars::prelude::*;
use std::ops::ControlFlow;
use chrono::Local;

struct Config {
    min_elo: i32,
    min_plys: usize,
    max_elo_diff: i32,
    min_base_time_s: i32,
    include_draws: bool,
    batch_size: usize,
}

const CONFIG: Config = Config {
    min_elo: 1800,
    min_plys: 20,
    max_elo_diff: 500,
    min_base_time_s: 300,
    include_draws: true,
    batch_size: 50_000,
};

#[derive(Default, Clone)]
struct GameData {
    white_elo: i32,
    black_elo: i32,
    result: String,
    moves: Vec<String>,
    opening: String,
}

struct FilteredVisitor {
    current_game: GameData,
    skip_game: bool,
    valid_time_control: bool,
    parsed_count: usize,
    accepted_count: usize,
}

impl FilteredVisitor {
    fn new() -> Self {
        Self {
            current_game: GameData::default(),
            skip_game: false,
            valid_time_control: false,
            parsed_count: 0,
            accepted_count: 0,
        }
    }

    fn log_status(&self) {
        let now = Local::now();
        println!("{} - INFO - Parsed {} games. Accepted {} so far.", 
            now.format("%Y-%m-%d %H:%M:%S,%3f"), 
            self.parsed_count, 
            self.accepted_count
        );
    }
}

impl Visitor for FilteredVisitor {
    type Tags = ();
    type Movetext = ();
    type Output = Option<GameData>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        self.parsed_count += 1;
        if self.parsed_count % 10000 == 0 {
            self.log_status();
        }

        self.current_game = GameData::default();
        self.skip_game = false;
        self.valid_time_control = false;
        ControlFlow::Continue(())
    }

    fn tag(&mut self, _tags: &mut Self::Tags, key: &[u8], value: RawTag<'_>) -> ControlFlow<Self::Output> {
        if self.skip_game { return ControlFlow::Continue(()); }

        let key_str = std::str::from_utf8(key).unwrap_or_default();
        let val_bytes = value.as_bytes();
        let val_str = std::str::from_utf8(val_bytes).unwrap_or_default().trim().trim_matches('"');

        match key_str {
            "WhiteElo" => self.current_game.white_elo = val_str.parse().unwrap_or(0),
            "BlackElo" => self.current_game.black_elo = val_str.parse().unwrap_or(0),
            "Result" => self.current_game.result = val_str.to_string(),
            "Opening" => self.current_game.opening = val_str.to_string(),
            "TimeControl" => {
                if let Some(base) = val_str.split('+').next() {
                    if let Ok(secs) = base.parse::<i32>() {
                        if secs >= CONFIG.min_base_time_s {
                            self.valid_time_control = true;
                        } else {
                            self.skip_game = true;
                        }
                    }
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, _tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        if self.skip_game { return ControlFlow::Break(None); }

        if !self.valid_time_control { return ControlFlow::Break(None); }

        if self.current_game.white_elo < CONFIG.min_elo || self.current_game.black_elo < CONFIG.min_elo {
            return ControlFlow::Break(None);
        }

        if (self.current_game.white_elo - self.current_game.black_elo).abs() > CONFIG.max_elo_diff {
            return ControlFlow::Break(None);
        }

        match self.current_game.result.as_str() {
            "1-0" | "0-1" => {}, 
            "1/2-1/2" => {
                if !CONFIG.include_draws { return ControlFlow::Break(None); }
            },
            _ => return ControlFlow::Break(None),
        }

        ControlFlow::Continue(())
    }

    fn san(&mut self, _movetext: &mut Self::Movetext, san_plus: SanPlus) -> ControlFlow<Self::Output> {
        self.current_game.moves.push(format!("{}", san_plus));
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {
        if self.current_game.moves.len() < CONFIG.min_plys {
            return None;
        }

        self.accepted_count += 1;
        let game = std::mem::take(&mut self.current_game);
        Some(game)
    }
}

fn main() -> Result<()> {
    let path = Path::new("data/download_links.txt");
    if !path.exists() {
        return Ok(());
    }
    
    let file = File::open(&path)?;
    let reader = BufReader::new(file);

    let mut grand_total_saved = 0;

    for line in reader.lines() {
        let link = line?;
        if link.trim().is_empty() { continue; }

        let now = Local::now();
        let date = link.split('_').last()
            .and_then(|s| s.split('.').next())
            .unwrap_or("unknown");
        println!("{} - INFO - Processing {}", now.format("%Y-%m-%d %H:%M:%S,%3f"), date);

        match process_link(&link) {
            Ok(count) => grand_total_saved += count,
            Err(e) => println!("{} - ERROR - Failed to process {}: {}", now.format("%Y-%m-%d %H:%M:%S,%3f"), date, e),
        }
    }

    let now = Local::now();
    println!("{} - INFO - All files processed. Grand total games saved: {}", 
        now.format("%Y-%m-%d %H:%M:%S,%3f"), 
        grand_total_saved
    );

    Ok(())
}

fn process_link(link: &str) -> Result<usize> {
    let date = link.split('_').last()
        .and_then(|s| s.split('.').next())
        .unwrap_or("unknown");
        
    let response = reqwest::blocking::get(link)?;
    let decoder = zstd::stream::read::Decoder::new(response)?;
    let reader = BufReader::with_capacity(64 * 1024, decoder); 
    
    let mut pgn_reader = Reader::new(reader);
    let mut visitor = FilteredVisitor::new();
    
    let mut games_batch = Vec::new();
    let mut total_saved = 0;
    let mut part_idx = 0;

    while let Some(game) = pgn_reader.read_game(&mut visitor)? {
        if let Some(valid_game) = game {
            games_batch.push(valid_game);
            
            if games_batch.len() >= CONFIG.batch_size {
                save_batch(&games_batch, date, part_idx)?;
                total_saved += games_batch.len();
                part_idx += 1;
                games_batch.clear();
                
                let now = Local::now();
                println!("{} - INFO - Saved {} games to data/parquet/processed_games_{}_part_{}.parquet. Total games for {}: {}", 
                    now.format("%Y-%m-%d %H:%M:%S,%3f"), 
                    CONFIG.batch_size,
                    date,
                    part_idx,
                    date,
                    total_saved
                );
            }
        }
    }

    if !games_batch.is_empty() {
        save_batch(&games_batch, date, part_idx)?;
        total_saved += games_batch.len();
        
        let now = Local::now();
        println!("{} - INFO - Saved {} games to data/parquet/processed_games_{}_part_{}.parquet. Total games for {}: {}", 
            now.format("%Y-%m-%d %H:%M:%S,%3f"), 
            games_batch.len(),
            date,
            part_idx,
            date,
            total_saved
        );
    }

    let now = Local::now();
    println!("{} - INFO - Finished processing {}. Total games saved: {}", 
        now.format("%Y-%m-%d %H:%M:%S,%3f"), 
        date, 
        total_saved
    );
    Ok(total_saved)
}

fn save_batch(games: &[GameData], date: &str, part_idx: usize) -> Result<()> {
    let height = games.len();
    let white_elos: Series = Series::new("white_elo".into(), games.iter().map(|g| g.white_elo).collect::<Vec<_>>());
    let black_elos: Series = Series::new("black_elo".into(), games.iter().map(|g| g.black_elo).collect::<Vec<_>>());
    let results: Series = Series::new("result".into(), games.iter().map(|g| g.result.as_str()).collect::<Vec<_>>());
    let openings: Series = Series::new("opening".into(), games.iter().map(|g| g.opening.as_str()).collect::<Vec<_>>());
    let moves: Series = Series::new("moves".into(), games.iter().map(|g| g.moves.join(" ")).collect::<Vec<_>>());

    let mut df = DataFrame::new(height, vec![
        white_elos.into(),
        black_elos.into(),
        results.into(),
        moves.into(),
        openings.into(),
    ])?;

    let filename = format!("data/parquet/processed_games_{}_part_{}.parquet", date, part_idx);
    let mut file = std::fs::File::create(&filename)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;

    Ok(())
}
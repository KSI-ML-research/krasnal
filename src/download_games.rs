mod fetcher;

use anyhow::Result;
use fetcher::{CONFIG, FilteredVisitor, Manifest, ManifestEntry, get_file_prefix, save_batch};
use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::Reader;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    std::fs::create_dir_all(CONFIG.output_dir)?;

    let manifest_path = CONFIG.manifest_path;
    let mut manifest = Manifest::load(manifest_path);

    let links_file = std::path::Path::new(CONFIG.links_path);
    if !links_file.exists() {
        println!("Links file not found at {}", CONFIG.links_path);
        return Ok(());
    }

    let mut links: Vec<String> = BufReader::new(File::open(links_file)?)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    // Sort to process links from oldest to newest
    links.sort();

    let mut total_games = 0;
    for entry in manifest.links.values() {
        total_games += entry.games_count;
    }

    println!("Initial total games: {}", total_games);
    if total_games >= CONFIG.target_games {
        println!("Target reached! ({}/{})", total_games, CONFIG.target_games);
        return Ok(());
    }

    let pb = ProgressBar::new(CONFIG.target_games as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
        .progress_chars("#>-"));
    pb.set_position(total_games as u64);

    for (i, link) in links.iter().enumerate() {
        if total_games >= CONFIG.target_games {
            pb.finish_with_message("Target games reached!");
            break;
        }

        if manifest
            .links
            .get(link)
            .is_some_and(|entry| entry.status == "completed")
        {
            continue;
        }

        pb.set_message(format!(
            "Processing {} ({}/{})",
            get_file_prefix(link),
            i + 1,
            links.len()
        ));

        match process_link(link, &mut total_games, &pb) {
            Ok((count, fully_completed)) => {
                if fully_completed {
                    manifest.links.insert(
                        link.clone(),
                        ManifestEntry {
                            status: "completed".to_string(),
                            games_count: count,
                        },
                    );
                    manifest.save(manifest_path)?;
                    pb.println(format!(
                        "✓ Finished: {} (+{} games)",
                        get_file_prefix(link),
                        count
                    ));
                } else {
                    pb.println(format!(
                        "! Limit reached at: {} (+{} games)",
                        get_file_prefix(link),
                        count
                    ));
                }
            }
            Err(e) => {
                pb.println(format!("✗ Error processing {}: {}", link, e));
            }
        }
    }

    pb.finish_with_message(format!("Done! Total games: {}", total_games));
    Ok(())
}

fn process_link(link: &str, total_games: &mut usize, pb: &ProgressBar) -> Result<(usize, bool)> {
    let response = reqwest::blocking::get(link)?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::with_capacity(512 * 1024, response))?;
    let mut pgn_reader = Reader::new(BufReader::with_capacity(256 * 1024, decoder));
    let mut visitor = FilteredVisitor::new();

    let prefix = get_file_prefix(link);
    let mut batch = Vec::with_capacity(CONFIG.batch_size);
    let mut games_in_link = 0;
    let mut part_idx = 0;
    let mut fully_read = true;

    while let Some(game) = pgn_reader.read_game(&mut visitor)? {
        if let Some(valid_game) = game {
            batch.push(valid_game);
            games_in_link += 1;
            *total_games += 1;
            pb.inc(1);

            if batch.len() >= CONFIG.batch_size {
                let filename =
                    format!("{}/{}_part_{}.parquet", CONFIG.output_dir, prefix, part_idx);
                save_batch(&batch, &filename)?;
                batch.clear();
                part_idx += 1;
            }

            if *total_games >= CONFIG.target_games {
                fully_read = false;
                break;
            }
        }
    }

    if !batch.is_empty() {
        let filename = format!("{}/{}_part_{}.parquet", CONFIG.output_dir, prefix, part_idx);
        save_batch(&batch, &filename)?;
    }

    Ok((games_in_link, fully_read))
}

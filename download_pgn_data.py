"""
Script để download PGN files từ pgnmentor.com
Downloads chess game data in PGN format for training
"""

import os
import urllib.request
import urllib.parse
import zipfile
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def download_file(url: str, output_path: str, timeout: int = 30) -> bool:
    """Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        
        # Create request with User-Agent header to avoid blocking
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        # Download with timeout
        with urllib.request.urlopen(req, timeout=timeout) as response:
            # Check if response is OK
            if response.status != 200:
                print(f"Error: HTTP {response.status} for {url}")
                return False
            
            # Save file
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
        
        # Check if file was downloaded and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size = os.path.getsize(output_path) / 1024  # Size in KB
            print(f"Downloaded to {output_path} ({file_size:.2f} KB)")
            return True
        else:
            print(f"Error: Downloaded file is empty or doesn't exist")
            return False
            
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason} for {url}")
        if e.code == 404:
            print(f"  File not found. Player name might be incorrect.")
        return False
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason} for {url}")
        return False
    except Exception as e:
        print(f"Error downloading {url}: {type(e).__name__}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_player_games(player_name: str, data_dir: str = "data") -> str:
    """Download PGN file for a specific player.
    
    Args:
        player_name: Name of the player (e.g., "Carlsen", "Anand")
        data_dir: Directory to save data
        
    Returns:
        Path to extracted PGN file or None if failed
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "zip"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "pgn"), exist_ok=True)
    
    # Clean player name - remove spaces and special characters for URL
    # But keep the original for file naming
    url_player_name = player_name.replace(" ", "").replace("-", "")
    
    # URL format: https://www.pgnmentor.com/players/PlayerName.zip
    # URL encode the player name
    url_player_name_encoded = urllib.parse.quote(url_player_name)
    url = f"https://www.pgnmentor.com/players/{url_player_name_encoded}.zip"
    
    zip_path = os.path.join(data_dir, "zip", f"{player_name.replace(' ', '_')}.zip")
    pgn_dir = os.path.join(data_dir, "pgn")
    
    # Check if already downloaded
    pgn_file = os.path.join(pgn_dir, f"{player_name.replace(' ', '_')}.pgn")
    if os.path.exists(pgn_file):
        print(f"  {player_name}: Already exists, skipping download")
        return pgn_file
    
    # Download zip file
    if not download_file(url, zip_path):
        # Try alternative URL format (with spaces as underscores)
        alt_url = f"https://www.pgnmentor.com/players/{player_name.replace(' ', '_')}.zip"
        if url != alt_url:
            print(f"  Trying alternative URL: {alt_url}")
            if not download_file(alt_url, zip_path):
                return None
        else:
            return None
    
    # Extract zip file
    if not extract_zip(zip_path, pgn_dir):
        return None
    
    # Find the PGN file - try multiple possible names
    possible_names = [
        f"{player_name}.pgn",
        f"{player_name.replace(' ', '_')}.pgn",
        f"{url_player_name}.pgn",
        f"{url_player_name_encoded}.pgn"
    ]
    
    for name in possible_names:
        pgn_file = os.path.join(pgn_dir, name)
        if os.path.exists(pgn_file):
            return pgn_file
    
    # Sometimes the file might have different name - find any PGN file in the extracted directory
    pgn_files = list(Path(pgn_dir).glob("*.pgn"))
    if pgn_files:
        # If multiple files, prefer one that matches player name
        for pgn_file in pgn_files:
            if player_name.lower().replace(" ", "") in str(pgn_file).lower():
                return str(pgn_file)
        return str(pgn_files[0])
    
    print(f"  Warning: No PGN file found after extracting {player_name}")
    return None


def download_multiple_players(player_names: list, data_dir: str = "data", max_workers: int = 5):
    """Download PGN files for multiple players in parallel.
    
    Args:
        player_names: List of player names to download
        data_dir: Directory to save data
        max_workers: Number of parallel downloads (default: 5)
    """
    downloaded_files = []
    failed_downloads = []
    lock = Lock()  # For thread-safe printing
    
    def download_with_status(player_name, index, total):
        """Download a single player and return status."""
        with lock:
            print(f"[{index}/{total}] Starting download: {player_name}...")
        
        pgn_file = download_player_games(player_name, data_dir)
        
        with lock:
            if pgn_file:
                print(f"[{index}/{total}] ✓ Successfully downloaded {player_name}")
                return (player_name, pgn_file, True)
            else:
                print(f"[{index}/{total}] ✗ Failed to download {player_name}")
                return (player_name, None, False)
    
    print(f"\nStarting parallel downloads (max {max_workers} concurrent)...")
    print("=" * 50)
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_player = {
            executor.submit(download_with_status, player_name, i+1, len(player_names)): player_name
            for i, player_name in enumerate(player_names)
        }
        
        # Process completed downloads
        for future in as_completed(future_to_player):
            player_name, pgn_file, success = future.result()
            if success:
                downloaded_files.append(pgn_file)
            else:
                failed_downloads.append(player_name)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Download Summary:")
    print(f"  Successful: {len(downloaded_files)}/{len(player_names)}")
    if failed_downloads:
        print(f"  Failed: {len(failed_downloads)}")
        print(f"  Failed players: {', '.join(failed_downloads[:10])}")
        if len(failed_downloads) > 10:
            print(f"  ... and {len(failed_downloads) - 10} more")
    
    return downloaded_files


def test_download(player_name: str = "Carlsen"):
    """Test download with a single player to check connectivity.
    
    Args:
        player_name: Name of player to test (default: "Carlsen")
    """
    print("Testing download connection...")
    print("=" * 50)
    result = download_player_games(player_name, "data")
    if result:
        print(f"\n✓ Test successful! Downloaded {player_name}")
        return True
    else:
        print(f"\n✗ Test failed for {player_name}")
        print("\nPossible issues:")
        print("  1. Internet connection problem")
        print("  2. pgnmentor.com might be down")
        print("  3. Player name might be incorrect")
        print("  4. Firewall/proxy blocking the connection")
        return False


def main():
    """Main function to download chess game data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download chess PGN files from pgnmentor.com')
    parser.add_argument('--test', action='store_true',
                       help='Test download with a single player (Carlsen)')
    parser.add_argument('--players', type=int, default=None,
                       help='Number of top players to download (default: all)')
    parser.add_argument('--parallel', type=int, default=5,
                       help='Number of parallel downloads (default: 5, max recommended: 10)')
    
    args = parser.parse_args()
    
    print("Chess PGN Data Downloader")
    print("=" * 50)
    
    # Test mode
    if args.test:
        test_download("Carlsen")
        return
    
    # List of top players to download (you can modify this list)
    # These are some of the strongest players with many games
    # Starting with a smaller, reliable list first
    top_players = [
        "Abdusattorov",    # Nodirbek Abdusattorov
        "Adams",           # Michael Adams
        "Akobian",         # Varuzhan Akobian
        "Akopian",         # Vladimir Akopian
        "Alburt",          # Lev Alburt
        "Alekhine",        # Alexander Alekhine
        "Alekseev",        # Evgeny Alekseev
        "Almasi",          # Zoltan Almasi
        "Anand",           # Viswanathan Anand
        "Anderssen",       # Adolf Anderssen
        "Andersson",       # Ulf Andersson
        "Andreikin",       # Dmitry Andreikin
        "Aronian",         # Levon Aronian
        "Ashley",          # Maurice Ashley
        "Averbakh",        # Yuri Averbakh
        "Azmaiparashvili", # Zurab Azmaiparashvili
        "Bacrot",          # Etienne Bacrot
        "Bareev",          # Evgeny Bareev
        "Becerra Rivero",  # Julio Becerra Rivero
        "Beliavsky",       # Alexander Beliavsky
        "Benjamin",        # Joel Benjamin
        "Benko",           # Pal Benko
        "Berliner",        # Hans Berliner
        "Bernstein",       # Ossip Bernstein
        "Bird",            # Henry Bird
        "Bisguier",        # Arthur Bisguier
        "Blackburne",      # Joseph Blackburne
        "Blatny",          # Pavel Blatny
        "Bogoljubow",      # Efim Bogoljubow
        "Boleslavsky",     # Isaac Boleslavsky
        "Bologan",         # Viktor Bologan
        "Botvinnik",       # Mikhail Botvinnik
        "Breyer",          # Gyula Breyer
        "Bronstein",       # David Bronstein
        "Browne",          # Walter Browne
        "Bruzon",          # Lazaro Bruzon
        "Bu",              # Bu Xiangzhi
        "Byrne",           # Robert Byrne
        "Capablanca",      # José Raúl Capablanca
        "Carlsen",         # Magnus Carlsen
        "Caruana",         # Fabiano Caruana
        "Chiburdanidze",   # Maia Chiburdanidze
        "Chigorin",        # Mikhail Chigorin
        "Christiansen",    # Larry Christiansen
        "DeFirmian",       # Nick DeFirmian
        "de LaBourdonnais", # Louis de LaBourdonnais
        "Denker",          # Arnold Denker
        "Ding",            # Ding Liren
        "Dominguez Perez", # Leinier Dominguez Perez
        "Dreev",           # Alexey Dreev
        "Duda",            # Jan-Krzysztof Duda
        "Dzindzichashvili", # Roman Dzindzichashvili
        "Ehlvest",         # Jaan Ehlvest
        "Eljanov",         # Pavel Eljanov
        "Erigaisi",        # Arjun Erigaisi
        "Euwe",            # Max Euwe
        "Evans",           # Larry Evans
        "Fedorowicz",      # John Fedorowicz
        "Fine",            # Reuben Fine
        "Finegold",        # Benjamin Finegold
        "Firouzja",        # Alireza Firouzja
        "Fischer",         # Robert James Fischer
        "Fishbein",        # Alexander Fishbein
        "Flohr",           # Salo Flohr
        "Gaprindashvili",  # Nona Gaprindashvili
        "Gashimov",        # Vugar Gashimov
        "Gelfand",         # Boris Gelfand
        "Geller",          # Efim Geller
        "Georgiev",        # Kiril Georgiev
        "Giri",            # Anish Giri
        "Gligoric",        # Svetozar Gligoric
        "Goldin",          # Alexander Goldin
        "Granda Zuniga",   # Julio Granda Zuniga
        "Grischuk",        # Alexander Grischuk
        "Gukesh",          # Dommaraju Gukesh
        "Gulko",           # Boris Gulko
        "Gunsberg",        # Isidor Gunsberg
        "Gurevich",        # Dmitry Gurevich
        "Gurevich",        # Mikhail Gurevich
        "Harikrishna",     # Pentala Harikrishna
        "Hort",            # Vlastimil Hort
        "Horwitz",         # Bernhard Horwitz
        "Hou",             # Hou Yifan
        "Huebner",         # Robert Huebner
        "Ibragimov",       # Ildar Ibragimov
        "Illescas Cordoba", # Miguel Illescas Cordoba
        "Inarkiev",        # Ernesto Inarkiev
        "Ivanchuk",        # Vassily Ivanchuk
        "Ivanov",          # Alexander Ivanov
        "Ivanov",          # Igor Ivanov
        "Ivkov",           # Borislav Ivkov
        "Jakovenko",       # Dmitry Jakovenko
        "Janowski",        # David Janowski
        "Jobava",          # Baadur Jobava
        "Jussupow",        # Artur Jussupow
        "Kaidanov",        # Gregory Kaidanov
        "Kamsky",          # Gata Kamsky
        "Karjakin",        # Sergey Karjakin
        "Karpov",          # Anatoly Karpov
        "Kasimdzhanov",    # Rustam Kasimdzhanov
        "Kasparov",        # Garry Kasparov
        "Kavalek",         # Lubomir Kavalek
        "Keres",           # Paul Keres
        "Keymer",          # Vincent Keymer
        "Khalifman",       # Alexander Khalifman
        "Kholmov",         # Ratmir Kholmov
        "Koneru",          # Koneru Humpy
        "Korchnoi",        # Viktor Korchnoi
        "Korobov",         # Anton Korobov
        "Kosteniuk",       # Alexandra Kosteniuk
        "Kotov",           # Alexander Kotov
        "Kramnik",         # Vladimir Kramnik
        "Krasenkow",       # Michal Krasenkow
        "Krush",           # Irina Krush
        "Kudrin",          # Sergey Kudrin
        "Lahno",           # Kateryna Lahno
        "Larsen",          # Bent Larsen
        "Lasker",          # Emanuel Lasker
        "Lautier",         # Joel Lautier
        "Le",              # Le Quang Liem
        "Leko",            # Peter Leko
        "Levenfish",       # Grigory Levenfish
        "Li",              # Li Chao
        "Lilienthal",      # Andre Lilienthal
        "Ljubojevic",      # Ljubomir Ljubojevic
        "Lputian",         # Smbat Lputian
        "MacKenzie",       # George MacKenzie
        "Malakhov",        # Vladimir Malakhov
        "Mamedyarov",      # Shakhriyar Mamedyarov
        "Maroczy",         # Geza Maroczy
        "Marshall",        # Frank Marshall
        "McDonnell",       # Alexander McDonnell
        "McShane",         # Luke McShane
        "Mecking",         # Henrique Mecking
        "Mikenas",         # Vladas Mikenas
        "Miles",           # Anthony Miles
        "Milov",           # Vadim Milov
        "Morozevich",      # Alexander Morozevich
        "Morphy",          # Paul Morphy
        "Motylev",         # Alexander Motylev
        "Movsesian",       # Sergei Movsesian
        "Muzychuk",        # Mariya Muzychuk
        "Najdorf",         # Miguel Najdorf
        "Najer",           # Evgeny Najer
        "Nakamura",        # Hikaru Nakamura
        "Navara",          # David Navara
        "Negi",            # Parimarjan Negi
        "Nepomniachtchi",  # Ian Nepomniachtchi
        "Ni",              # Ni Hua
        "Nielsen",         # Peter Nielsen
        "Nikolic",         # Predrag Nikolic
        "Nimzowitsch",     # Aron Nimzowitsch
        "Nisipeanu",       # Liviu-Dieter Nisipeanu
        "Novikov",         # Igor Novikov
        "Nunn",            # John Nunn
        "Olafsson",        # Fridrik Olafsson
        "Oll",             # Lembit Oll
        "Onischuk",        # Alexander Onischuk
        "Pachman",         # Ludek Pachman
        "Paehtz",          # Elisabeth Paehtz
        "Panno",           # Oscar Panno
        "Paulsen",         # Louis Paulsen
        "Petrosian",       # Tigran Petrosian
        "Philidor",        # Francois Philidor
        "Pillsbury",       # Harry Pillsbury
        "Pilnik",          # Herman Pilnik
        "Polgar",          # Judit Polgar
        "Polgar",          # Sofia Polgar
        "Polgar",          # Zsuzsa Polgar
        "Polugaevsky",     # Lev Polugaevsky
        "Ponomariov",      # Ruslan Ponomariov
        "Portisch",        # Lajos Portisch
        "Praggnanandhaa",  # Rameshbabu Praggnanandhaa
        "Psakhis",         # Lev Psakhis
        "Quinteros",       # Miguel Quinteros
        "Radjabov",        # Teimour Radjabov
        "Rapport",         # Richard Rapport
        "Reshevsky",       # Samuel Reshevsky
        "Reti",            # Richard Reti
        "Ribli",           # Zoltan Ribli
        "Rohde",           # Michael Rohde
        "Rubinstein",      # Akiba Rubinstein
        "Rublevsky",       # Sergei Rublevsky
        "Saemisch",        # Friedrich Saemisch
        "Sakaev",          # Konstantin Sakaev
        "Salov",           # Valery Salov
        "Sasikiran",       # Krishnan Sasikiran
        "Schlechter",      # Carl Schlechter
        "Seirawan",        # Yasser Seirawan
        "Serper",          # Gregory Serper
        "Shabalov",        # Alexander Shabalov
        "Shamkovich",      # Leonid Shamkovich
        "Shirov",          # Alexei Shirov
        "Short",           # Nigel Short
        "Shulman",         # Yury Shulman
        "Smirin",          # Ilia Smirin
        "Smyslov",         # Vasily Smyslov
        "So",              # Wesley So
        "Sokolov",         # Ivan Sokolov
        "Soltis",          # Andrew Soltis
        "Spassky",         # Boris Spassky
        "Speelman",        # Jonathan Speelman
        "Spielmann",       # Rudolf Spielmann
        "Stahlberg",       # Gideon Stahlberg
        "Staunton",        # Howard Staunton
        "Stefanova",       # Antoaneta Stefanova
        "Stein",           # Leonid Stein
        "Steinitz",        # William Steinitz
        "Suetin",          # Alexey Suetin
        "Sultan Khan",     # Mir Sultan Khan
        "Sutovsky",        # Emil Sutovsky
        "Svidler",         # Peter Svidler
        "Szabo",           # Laszlo Szabo
        "Taimanov",        # Mark Taimanov
        "Tal",             # Mikhail Tal
        "Tarrasch",        # Siegbert Tarrasch
        "Tartakower",      # Savielly Tartakower
        "Teichmann",       # Richard Teichmann
        "Timman",          # Jan Timman
        "Tiviakov",        # Sergei Tiviakov
        "Tkachiev",        # Vladislav Tkachiev
        "Tomashevsky",     # Evgeny Tomashevsky
        "Topalov",         # Veselin Topalov
        "Torre Repetto",   # Carlos Torre Repetto
        "Uhlmann",         # Wolfgang Uhlmann
        "Unzicker",        # Wolfgang Unzicker
        "Ushenina",        # Anna Ushenina
        "Vachier-Lagrave", # Maxime Vachier-Lagrave
        "Vaganian",        # Rafael Vaganian
        "Vallejo Pons",    # Francisco Vallejo Pons
        "Van Wely",        # Loek Van Wely
        "Vitiugov",        # Nikita Vitiugov
        "Volokitin",       # Andrei Volokitin
        "Waitzkin",        # Joshua Waitzkin
        "Wang",            # Wang Yue
        "Wang",            # Wang Hao
        "Wei",             # Wei Yi
        "Winawer",         # Simon Winawer
        "Wojtaszek",       # Radoslaw Wojtaszek
        "Wojtkiewicz",     # Aleksander Wojtkiewicz
        "Wolff",           # Patrick Wolff
        "Xie",             # Xie Jun
        "Xu",              # Xu Yuhua
        "Ye",              # Ye Jiangchuan
        "Yermolinsky",     # Alex Yermolinsky
        "Yu",              # Yu Yangyi
        "Yudasin",         # Leonid Yudasin
        "Zhu",             # Zhu Chen
        "Zukertort",       # Johannes Zukertort
        "Zvjaginsev",      # Vadim Zvjaginsev
    ]
    
    # Limit number of players if specified
    if args.players:
        top_players = top_players[:args.players]
    
    print(f"Will download games from {len(top_players)} players")
    print("First 10 players:", ", ".join(top_players[:10]))
    if len(top_players) > 10:
        print(f"... and {len(top_players) - 10} more")
    
    # Test connection first
    print("\nTesting connection...")
    if not test_download("Carlsen"):
        print("\n⚠ Warning: Test download failed!")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Download all players
    print("\n" + "=" * 50)
    print(f"Starting parallel downloads (max {args.parallel} concurrent)...")
    downloaded_files = download_multiple_players(
        top_players,
        data_dir="data",
        max_workers=args.parallel
    )
    
    print("\n" + "=" * 50)
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(downloaded_files)}/{len(top_players)} PGN files")
    print(f"Files saved in: data/pgn/")
    
    if downloaded_files:
        print("\nDownloaded files:")
        total_size = 0
        for file in downloaded_files:
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            total_size += file_size
            print(f"  ✓ {os.path.basename(file)} ({file_size:.2f} MB)")
        print(f"\nTotal size: {total_size:.2f} MB")
    else:
        print("\n⚠ No files were downloaded. Please check:")
        print("  1. Internet connection")
        print("  2. Try running with --test flag first")
        print("  3. Check if pgnmentor.com is accessible")


if __name__ == "__main__":
    main()

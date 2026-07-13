use std::fmt::{Display, Formatter};
use std::str::FromStr;

pub const CLASSIC_STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub type Bitboard = u64;

const FILE_A: Bitboard = 0x0101_0101_0101_0101;
const FILE_H: Bitboard = 0x8080_8080_8080_8080;
const RANK_1: Bitboard = 0x0000_0000_0000_00ff;
const RANK_3: Bitboard = 0x0000_0000_00ff_0000;
const RANK_6: Bitboard = 0x0000_ff00_0000_0000;
const RANK_8: Bitboard = 0xff00_0000_0000_0000;

const WHITE_KINGSIDE: u8 = 0b0001;
const WHITE_QUEENSIDE: u8 = 0b0010;
const BLACK_KINGSIDE: u8 = 0b0100;
const BLACK_QUEENSIDE: u8 = 0b1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    pub fn pawn_direction(self) -> i8 {
        match self {
            Self::White => 1,
            Self::Black => -1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    pub fn from_fen(ch: char) -> Option<Self> {
        let color = if ch.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let kind = match ch.to_ascii_lowercase() {
            'p' => PieceKind::Pawn,
            'n' => PieceKind::Knight,
            'b' => PieceKind::Bishop,
            'r' => PieceKind::Rook,
            'q' => PieceKind::Queen,
            'k' => PieceKind::King,
            _ => return None,
        };
        Some(Self { color, kind })
    }

    pub fn fen_char(self) -> char {
        let base = match self.kind {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        };
        match self.color {
            Color::White => base.to_ascii_uppercase(),
            Color::Black => base,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

impl Square {
    pub fn from_file_rank(file: u8, rank: u8) -> Option<Self> {
        if file < 8 && rank < 8 {
            Some(Self(rank * 8 + file))
        } else {
            None
        }
    }

    pub fn file(self) -> u8 {
        self.0 % 8
    }

    pub fn rank(self) -> u8 {
        self.0 / 8
    }

    pub fn offset(self, df: i8, dr: i8) -> Option<Self> {
        let file = self.file() as i8 + df;
        let rank = self.rank() as i8 + dr;
        if (0..8).contains(&file) && (0..8).contains(&rank) {
            Some(Self((rank as u8) * 8 + file as u8))
        } else {
            None
        }
    }

    pub fn to_coord(self) -> String {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        format!("{file}{rank}")
    }
}

impl Display for Square {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_coord())
    }
}

impl FromStr for Square {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err("square must be exactly two characters".to_string());
        }
        if !(b'a'..=b'h').contains(&bytes[0]) || !(b'1'..=b'8').contains(&bytes[1]) {
            return Err(format!("invalid square: {s}"));
        }
        Ok(Self((bytes[1] - b'1') * 8 + (bytes[0] - b'a')))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChessMove {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
}

impl ChessMove {
    pub fn new(from: Square, to: Square, promotion: Option<PieceKind>) -> Self {
        Self {
            from,
            to,
            promotion,
        }
    }

    pub fn from_uci(input: &str) -> Result<Self, String> {
        if input.len() < 4 || input.len() > 5 {
            return Err(format!("invalid UCI move: {input}"));
        }
        let from = Square::from_str(&input[..2])?;
        let to = Square::from_str(&input[2..4])?;
        let promotion = input
            .as_bytes()
            .get(4)
            .map(|ch| match ch.to_ascii_lowercase() {
                b'n' => Ok(PieceKind::Knight),
                b'b' => Ok(PieceKind::Bishop),
                b'r' => Ok(PieceKind::Rook),
                b'q' => Ok(PieceKind::Queen),
                _ => Err(format!("invalid promotion in move: {input}")),
            })
            .transpose()?;
        Ok(Self::new(from, to, promotion))
    }

    pub fn to_uci(self) -> String {
        let mut out = format!("{}{}", self.from, self.to);
        if let Some(promo) = self.promotion {
            let ch = match promo {
                PieceKind::Knight => 'n',
                PieceKind::Bishop => 'b',
                PieceKind::Rook => 'r',
                PieceKind::Queen => 'q',
                PieceKind::Pawn | PieceKind::King => 'q',
            };
            out.push(ch);
        }
        out
    }
}

impl Display for ChessMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_uci())
    }
}

#[derive(Clone, Debug)]
pub struct UndoState {
    moved_piece: Piece,
    captured_piece: Option<(Square, Piece)>,
    previous_castling: u8,
    previous_en_passant: Option<Square>,
    previous_halfmove_clock: u32,
    previous_fullmove_number: u32,
    previous_position_hash: u64,
}

#[derive(Clone, Debug)]
pub struct Board {
    squares: [Option<Piece>; 64],
    bitboards: [Bitboard; 12],
    pub side_to_move: Color,
    castling_rights: u8,
    en_passant: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
    position_hash: u64,
    repetition_history: Vec<u64>,
}

impl Default for Board {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Board {
    pub fn startpos() -> Self {
        Self::from_fen(CLASSIC_STARTPOS_FEN).expect("valid classic start position")
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut parts = fen.split_whitespace();
        let board_part = parts
            .next()
            .ok_or_else(|| "FEN is missing board definition".to_string())?;
        let side_part = parts
            .next()
            .ok_or_else(|| "FEN is missing side to move".to_string())?;
        let castling_part = parts
            .next()
            .ok_or_else(|| "FEN is missing castling rights".to_string())?;
        let en_passant_part = parts
            .next()
            .ok_or_else(|| "FEN is missing en passant square".to_string())?;
        let halfmove_part = parts.next().unwrap_or("0");
        let fullmove_part = parts.next().unwrap_or("1");

        let mut squares = [None; 64];
        let ranks: Vec<&str> = board_part.split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN board definition must contain 8 ranks".to_string());
        }

        for (fen_rank, rank_text) in ranks.iter().enumerate() {
            let mut file = 0u8;
            let board_rank = 7u8.saturating_sub(fen_rank as u8);
            for ch in rank_text.chars() {
                if ch.is_ascii_digit() {
                    file = file
                        .checked_add(ch.to_digit(10).unwrap_or(0) as u8)
                        .ok_or_else(|| "FEN rank overflow".to_string())?;
                } else {
                    let piece = Piece::from_fen(ch)
                        .ok_or_else(|| format!("invalid piece character in FEN: {ch}"))?;
                    let square = Square::from_file_rank(file, board_rank)
                        .ok_or_else(|| "invalid square while parsing FEN".to_string())?;
                    squares[square.0 as usize] = Some(piece);
                    file += 1;
                }
            }
            if file != 8 {
                return Err("each FEN rank must describe exactly 8 files".to_string());
            }
        }

        let side_to_move = match side_part {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("invalid side to move: {other}")),
        };

        let mut castling_rights = 0u8;
        if castling_part != "-" {
            for ch in castling_part.chars() {
                match ch {
                    'K' => castling_rights |= WHITE_KINGSIDE,
                    'Q' => castling_rights |= WHITE_QUEENSIDE,
                    'k' => castling_rights |= BLACK_KINGSIDE,
                    'q' => castling_rights |= BLACK_QUEENSIDE,
                    _ => return Err(format!("invalid castling right: {ch}")),
                }
            }
        }

        let en_passant = if en_passant_part == "-" {
            None
        } else {
            Some(Square::from_str(en_passant_part)?)
        };

        let halfmove_clock = halfmove_part
            .parse::<u32>()
            .map_err(|_| "invalid halfmove clock".to_string())?;
        let fullmove_number = fullmove_part
            .parse::<u32>()
            .map_err(|_| "invalid fullmove number".to_string())?;

        let mut board = Self {
            squares,
            bitboards: [0; 12],
            side_to_move,
            castling_rights,
            en_passant,
            halfmove_clock,
            fullmove_number,
            position_hash: 0,
            repetition_history: Vec::new(),
        };
        board.rebuild_bitboards();
        board.position_hash = board.recompute_position_hash();
        board.repetition_history.push(board.position_hash);
        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut board = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let square = Square(rank * 8 + file);
                match self.piece_at(square) {
                    Some(piece) => {
                        if empty > 0 {
                            board.push(char::from(b'0' + empty));
                            empty = 0;
                        }
                        board.push(piece.fen_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                board.push(char::from(b'0' + empty));
            }
            if rank > 0 {
                board.push('/');
            }
        }

        let side = match self.side_to_move {
            Color::White => "w",
            Color::Black => "b",
        };

        let mut castling = String::new();
        if self.castling_rights & WHITE_KINGSIDE != 0 {
            castling.push('K');
        }
        if self.castling_rights & WHITE_QUEENSIDE != 0 {
            castling.push('Q');
        }
        if self.castling_rights & BLACK_KINGSIDE != 0 {
            castling.push('k');
        }
        if self.castling_rights & BLACK_QUEENSIDE != 0 {
            castling.push('q');
        }
        if castling.is_empty() {
            castling.push('-');
        }

        let en_passant = self
            .en_passant
            .map(|sq| sq.to_string())
            .unwrap_or_else(|| "-".to_string());

        format!(
            "{board} {side} {castling} {en_passant} {} {}",
            self.halfmove_clock, self.fullmove_number
        )
    }

    pub fn position_signature(&self) -> String {
        let fen = self.to_fen();
        let mut parts = fen.split_whitespace();
        let board = parts.next().unwrap_or_default();
        let side = parts.next().unwrap_or_default();
        let castling = parts.next().unwrap_or_default();
        let en_passant = parts.next().unwrap_or_default();
        format!("{board} {side} {castling} {en_passant}")
    }

    pub fn position_hash(&self) -> u64 {
        self.position_hash
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    pub fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    pub fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.squares[square.0 as usize]
    }

    pub fn pieces(&self, color: Color, kind: PieceKind) -> Bitboard {
        self.bitboards[piece_index(Piece { color, kind })]
    }

    pub fn occupancy(&self, color: Color) -> Bitboard {
        let offset = match color {
            Color::White => 0,
            Color::Black => 6,
        };
        self.bitboards[offset..offset + 6]
            .iter()
            .fold(0, |occupancy, pieces| occupancy | pieces)
    }

    pub fn set_piece_at(&mut self, square: Square, piece: Option<Piece>) {
        if let Some(previous) = self.squares[square.0 as usize] {
            self.position_hash ^= piece_hash(previous, square);
            self.bitboards[piece_index(previous)] ^= 1_u64 << square.0;
        }
        self.squares[square.0 as usize] = piece;
        if let Some(piece) = piece {
            self.position_hash ^= piece_hash(piece, square);
            self.bitboards[piece_index(piece)] ^= 1_u64 << square.0;
        }
    }

    pub fn parse_uci_move(&self, text: &str) -> Result<ChessMove, String> {
        let parsed = ChessMove::from_uci(text)?;
        let mut clone = self.clone();
        if clone
            .generate_legal_moves()
            .into_iter()
            .any(|mv| mv == parsed)
        {
            Ok(parsed)
        } else {
            Err(format!("illegal move for current position: {text}"))
        }
    }

    pub fn is_threefold_repetition(&self) -> bool {
        let current = self.position_hash;
        self.repetition_history
            .iter()
            .filter(|hash| **hash == current)
            .count()
            >= 3
    }

    pub fn king_square(&self, color: Color) -> Option<Square> {
        self.squares
            .iter()
            .enumerate()
            .find_map(|(idx, piece)| match piece {
                Some(Piece {
                    color: piece_color,
                    kind: PieceKind::King,
                }) if *piece_color == color => Some(Square(idx as u8)),
                _ => None,
            })
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.king_square(color)
            .is_some_and(|king_sq| self.is_square_attacked(king_sq, color.opposite()))
    }

    pub fn is_square_attacked(&self, square: Square, by: Color) -> bool {
        let pawn_rank_delta = match by {
            Color::White => -1,
            Color::Black => 1,
        };
        for df in [-1, 1] {
            if let Some(src) = square.offset(df, pawn_rank_delta)
                && self.piece_at(src)
                    == Some(Piece {
                        color: by,
                        kind: PieceKind::Pawn,
                    })
            {
                return true;
            }
        }

        for (df, dr) in [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ] {
            if let Some(src) = square.offset(df, dr)
                && self.piece_at(src)
                    == Some(Piece {
                        color: by,
                        kind: PieceKind::Knight,
                    })
            {
                return true;
            }
        }

        for (df, dr) in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ] {
            if let Some(src) = square.offset(df, dr)
                && self.piece_at(src)
                    == Some(Piece {
                        color: by,
                        kind: PieceKind::King,
                    })
            {
                return true;
            }
        }

        self.slider_attack(
            square,
            by,
            &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
            &[PieceKind::Bishop, PieceKind::Queen],
        ) || self.slider_attack(
            square,
            by,
            &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            &[PieceKind::Rook, PieceKind::Queen],
        )
    }

    fn slider_attack(
        &self,
        square: Square,
        by: Color,
        directions: &[(i8, i8)],
        kinds: &[PieceKind],
    ) -> bool {
        for &(df, dr) in directions {
            let mut current = square;
            while let Some(next) = current.offset(df, dr) {
                current = next;
                if let Some(piece) = self.piece_at(current) {
                    if piece.color == by && kinds.contains(&piece.kind) {
                        return true;
                    }
                    break;
                }
            }
        }
        false
    }

    pub fn generate_legal_moves(&mut self) -> Vec<ChessMove> {
        let side = self.side_to_move;
        let moves = self.generate_pseudo_legal_moves();
        let mut legal = Vec::with_capacity(moves.len());
        for mv in moves {
            if let Ok(undo) = self.make_move(mv) {
                if !self.in_check(side) {
                    legal.push(mv);
                }
                self.unmake_move(mv, undo);
            }
        }
        legal
    }

    pub fn generate_capture_moves(&mut self) -> Vec<ChessMove> {
        self.generate_legal_moves()
            .into_iter()
            .filter(|mv| {
                self.piece_at(mv.to).is_some()
                    || self.piece_at(mv.from).is_some_and(|piece| {
                        piece.kind == PieceKind::Pawn && self.en_passant == Some(mv.to)
                    })
            })
            .collect()
    }

    fn generate_pseudo_legal_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::new();
        let color = self.side_to_move;

        self.generate_bitboard_pawn_moves(color, &mut moves);
        self.generate_bitboard_step_moves(
            self.pieces(color, PieceKind::Knight),
            color,
            knight_attacks,
            &mut moves,
        );
        let occupied = self.occupancy(Color::White) | self.occupancy(Color::Black);
        self.generate_bitboard_slider_moves(
            self.pieces(color, PieceKind::Bishop),
            color,
            occupied,
            bishop_attacks,
            &mut moves,
        );
        self.generate_bitboard_slider_moves(
            self.pieces(color, PieceKind::Rook),
            color,
            occupied,
            rook_attacks,
            &mut moves,
        );
        self.generate_bitboard_slider_moves(
            self.pieces(color, PieceKind::Queen),
            color,
            occupied,
            queen_attacks,
            &mut moves,
        );
        let kings = self.pieces(color, PieceKind::King);
        self.generate_bitboard_step_moves(kings, color, king_attacks, &mut moves);
        for_each_bit(kings, |from| {
            self.generate_castling_moves(from, color, &mut moves)
        });

        moves
    }

    fn generate_bitboard_step_moves(
        &self,
        mut pieces: Bitboard,
        color: Color,
        attacks: fn(Square) -> Bitboard,
        moves: &mut Vec<ChessMove>,
    ) {
        let own_occupancy = self.occupancy(color);
        while pieces != 0 {
            let from = pop_lsb(&mut pieces);
            let mut targets = attacks(from) & !own_occupancy;
            while targets != 0 {
                moves.push(ChessMove::new(from, pop_lsb(&mut targets), None));
            }
        }
    }

    fn generate_bitboard_pawn_moves(&self, color: Color, moves: &mut Vec<ChessMove>) {
        let pawns = self.pieces(color, PieceKind::Pawn);
        let occupied = self.occupancy(Color::White) | self.occupancy(Color::Black);
        let enemy_occupancy = self.occupancy(color.opposite());

        match color {
            Color::White => {
                let single_pushes = white_single_pawn_pushes(pawns, occupied);
                append_pawn_moves(single_pushes & !RANK_8, -8, moves);
                append_pawn_promotions(single_pushes & RANK_8, -8, moves);

                let double_pushes = (single_pushes & (RANK_3)) << 8 & !occupied;
                append_pawn_moves(double_pushes, -16, moves);

                let left_captures = ((pawns & !FILE_A) << 7) & enemy_occupancy;
                let right_captures = ((pawns & !FILE_H) << 9) & enemy_occupancy;
                append_pawn_moves(left_captures & !RANK_8, -7, moves);
                append_pawn_moves(right_captures & !RANK_8, -9, moves);
                append_pawn_promotions(left_captures & RANK_8, -7, moves);
                append_pawn_promotions(right_captures & RANK_8, -9, moves);
            }
            Color::Black => {
                let single_pushes = black_single_pawn_pushes(pawns, occupied);
                append_pawn_moves(single_pushes & !RANK_1, 8, moves);
                append_pawn_promotions(single_pushes & RANK_1, 8, moves);

                let double_pushes = (single_pushes & RANK_6) >> 8 & !occupied;
                append_pawn_moves(double_pushes, 16, moves);

                let left_captures = ((pawns & !FILE_A) >> 9) & enemy_occupancy;
                let right_captures = ((pawns & !FILE_H) >> 7) & enemy_occupancy;
                append_pawn_moves(left_captures & !RANK_1, 9, moves);
                append_pawn_moves(right_captures & !RANK_1, 7, moves);
                append_pawn_promotions(left_captures & RANK_1, 9, moves);
                append_pawn_promotions(right_captures & RANK_1, 7, moves);
            }
        }

        if let Some(target) = self.en_passant {
            for file_delta in [-1, 1] {
                if let Some(from) = target.offset(-file_delta, -color.pawn_direction())
                    && pawns & (1_u64 << from.0) != 0
                {
                    moves.push(ChessMove::new(from, target, None));
                }
            }
        }
    }

    fn generate_bitboard_slider_moves(
        &self,
        mut pieces: Bitboard,
        color: Color,
        occupied: Bitboard,
        attacks: fn(Square, Bitboard) -> Bitboard,
        moves: &mut Vec<ChessMove>,
    ) {
        let own_occupancy = self.occupancy(color);
        while pieces != 0 {
            let from = pop_lsb(&mut pieces);
            let mut targets = attacks(from, occupied) & !own_occupancy;
            while targets != 0 {
                moves.push(ChessMove::new(from, pop_lsb(&mut targets), None));
            }
        }
    }

    fn generate_castling_moves(&self, from: Square, color: Color, moves: &mut Vec<ChessMove>) {
        if self.in_check(color) {
            return;
        }
        match color {
            Color::White => {
                if self.castling_rights & WHITE_KINGSIDE != 0
                    && self
                        .piece_at(Square::from_str("f1").expect("valid"))
                        .is_none()
                    && self
                        .piece_at(Square::from_str("g1").expect("valid"))
                        .is_none()
                    && !self
                        .is_square_attacked(Square::from_str("f1").expect("valid"), Color::Black)
                    && !self
                        .is_square_attacked(Square::from_str("g1").expect("valid"), Color::Black)
                {
                    moves.push(ChessMove::new(
                        from,
                        Square::from_str("g1").expect("valid"),
                        None,
                    ));
                }
                if self.castling_rights & WHITE_QUEENSIDE != 0
                    && self
                        .piece_at(Square::from_str("b1").expect("valid"))
                        .is_none()
                    && self
                        .piece_at(Square::from_str("c1").expect("valid"))
                        .is_none()
                    && self
                        .piece_at(Square::from_str("d1").expect("valid"))
                        .is_none()
                    && !self
                        .is_square_attacked(Square::from_str("c1").expect("valid"), Color::Black)
                    && !self
                        .is_square_attacked(Square::from_str("d1").expect("valid"), Color::Black)
                {
                    moves.push(ChessMove::new(
                        from,
                        Square::from_str("c1").expect("valid"),
                        None,
                    ));
                }
            }
            Color::Black => {
                if self.castling_rights & BLACK_KINGSIDE != 0
                    && self
                        .piece_at(Square::from_str("f8").expect("valid"))
                        .is_none()
                    && self
                        .piece_at(Square::from_str("g8").expect("valid"))
                        .is_none()
                    && !self
                        .is_square_attacked(Square::from_str("f8").expect("valid"), Color::White)
                    && !self
                        .is_square_attacked(Square::from_str("g8").expect("valid"), Color::White)
                {
                    moves.push(ChessMove::new(
                        from,
                        Square::from_str("g8").expect("valid"),
                        None,
                    ));
                }
                if self.castling_rights & BLACK_QUEENSIDE != 0
                    && self
                        .piece_at(Square::from_str("b8").expect("valid"))
                        .is_none()
                    && self
                        .piece_at(Square::from_str("c8").expect("valid"))
                        .is_none()
                    && self
                        .piece_at(Square::from_str("d8").expect("valid"))
                        .is_none()
                    && !self
                        .is_square_attacked(Square::from_str("c8").expect("valid"), Color::White)
                    && !self
                        .is_square_attacked(Square::from_str("d8").expect("valid"), Color::White)
                {
                    moves.push(ChessMove::new(
                        from,
                        Square::from_str("c8").expect("valid"),
                        None,
                    ));
                }
            }
        }
    }

    pub fn make_move(&mut self, mv: ChessMove) -> Result<UndoState, String> {
        let moved_piece = self
            .piece_at(mv.from)
            .ok_or_else(|| format!("no piece on {}", mv.from))?;
        if moved_piece.color != self.side_to_move {
            return Err("attempted to move the wrong side".to_string());
        }

        let mut captured_piece = self.piece_at(mv.to).map(|piece| (mv.to, piece));
        if moved_piece.kind == PieceKind::Pawn
            && self.en_passant == Some(mv.to)
            && self.piece_at(mv.to).is_none()
        {
            let capture_square = Square::from_file_rank(mv.to.file(), mv.from.rank())
                .ok_or_else(|| "invalid en passant capture square".to_string())?;
            captured_piece = self
                .piece_at(capture_square)
                .map(|piece| (capture_square, piece));
        }

        if captured_piece.is_some_and(|(_, piece)| piece.kind == PieceKind::King) {
            if moved_piece.kind == PieceKind::Pawn
                && self.en_passant == Some(mv.to)
                && self.piece_at(mv.to).is_none()
            {
                if let Some(capture_square) = Square::from_file_rank(mv.to.file(), mv.from.rank()) {
                    if let Some((_, piece)) = captured_piece {
                        self.set_piece_at(capture_square, Some(piece));
                    }
                }
            }
            return Err("king capture is not a legal chess move".to_string());
        }

        let undo = UndoState {
            moved_piece,
            captured_piece,
            previous_castling: self.castling_rights,
            previous_en_passant: self.en_passant,
            previous_halfmove_clock: self.halfmove_clock,
            previous_fullmove_number: self.fullmove_number,
            previous_position_hash: self.position_hash,
        };

        if let Some((capture_square, _)) = captured_piece
            && moved_piece.kind == PieceKind::Pawn
            && self.en_passant == Some(mv.to)
            && self.piece_at(mv.to).is_none()
        {
            self.set_piece_at(capture_square, None);
        }

        self.set_piece_at(mv.from, None);

        let mut placed_piece = moved_piece;
        if let Some(promotion) = mv.promotion {
            placed_piece.kind = promotion;
        }
        self.set_piece_at(mv.to, Some(placed_piece));

        if moved_piece.kind == PieceKind::King {
            match moved_piece.color {
                Color::White => self.set_castling_rights(
                    self.castling_rights & !(WHITE_KINGSIDE | WHITE_QUEENSIDE),
                ),
                Color::Black => self.set_castling_rights(
                    self.castling_rights & !(BLACK_KINGSIDE | BLACK_QUEENSIDE),
                ),
            }

            match (mv.from.to_coord().as_str(), mv.to.to_coord().as_str()) {
                ("e1", "g1") => {
                    self.set_piece_at(Square::from_str("h1").expect("valid"), None);
                    self.set_piece_at(
                        Square::from_str("f1").expect("valid"),
                        Some(Piece {
                            color: Color::White,
                            kind: PieceKind::Rook,
                        }),
                    );
                }
                ("e1", "c1") => {
                    self.set_piece_at(Square::from_str("a1").expect("valid"), None);
                    self.set_piece_at(
                        Square::from_str("d1").expect("valid"),
                        Some(Piece {
                            color: Color::White,
                            kind: PieceKind::Rook,
                        }),
                    );
                }
                ("e8", "g8") => {
                    self.set_piece_at(Square::from_str("h8").expect("valid"), None);
                    self.set_piece_at(
                        Square::from_str("f8").expect("valid"),
                        Some(Piece {
                            color: Color::Black,
                            kind: PieceKind::Rook,
                        }),
                    );
                }
                ("e8", "c8") => {
                    self.set_piece_at(Square::from_str("a8").expect("valid"), None);
                    self.set_piece_at(
                        Square::from_str("d8").expect("valid"),
                        Some(Piece {
                            color: Color::Black,
                            kind: PieceKind::Rook,
                        }),
                    );
                }
                _ => {}
            }
        }

        if moved_piece.kind == PieceKind::Rook {
            match mv.from.to_coord().as_str() {
                "a1" => self.set_castling_rights(self.castling_rights & !WHITE_QUEENSIDE),
                "h1" => self.set_castling_rights(self.castling_rights & !WHITE_KINGSIDE),
                "a8" => self.set_castling_rights(self.castling_rights & !BLACK_QUEENSIDE),
                "h8" => self.set_castling_rights(self.castling_rights & !BLACK_KINGSIDE),
                _ => {}
            }
        }

        if let Some((captured_square, _captured)) = captured_piece {
            match captured_square.to_coord().as_str() {
                "a1" => self.set_castling_rights(self.castling_rights & !WHITE_QUEENSIDE),
                "h1" => self.set_castling_rights(self.castling_rights & !WHITE_KINGSIDE),
                "a8" => self.set_castling_rights(self.castling_rights & !BLACK_QUEENSIDE),
                "h8" => self.set_castling_rights(self.castling_rights & !BLACK_KINGSIDE),
                _ => {}
            }
        }

        self.set_en_passant(None);
        if moved_piece.kind == PieceKind::Pawn && mv.from.rank().abs_diff(mv.to.rank()) == 2 {
            self.set_en_passant(Square::from_file_rank(
                mv.from.file(),
                (mv.from.rank() + mv.to.rank()) / 2,
            ));
        }

        self.halfmove_clock = if moved_piece.kind == PieceKind::Pawn || captured_piece.is_some() {
            0
        } else {
            self.halfmove_clock + 1
        };

        if self.side_to_move == Color::Black {
            self.fullmove_number += 1;
        }
        self.set_side_to_move(self.side_to_move.opposite());
        self.repetition_history.push(self.position_hash);
        Ok(undo)
    }

    pub fn unmake_move(&mut self, mv: ChessMove, undo: UndoState) {
        let _ = self.repetition_history.pop();
        self.set_side_to_move(self.side_to_move.opposite());
        self.set_castling_rights(undo.previous_castling);
        self.set_en_passant(undo.previous_en_passant);
        self.halfmove_clock = undo.previous_halfmove_clock;
        self.fullmove_number = undo.previous_fullmove_number;

        self.set_piece_at(mv.from, Some(undo.moved_piece));
        self.set_piece_at(mv.to, None);

        if undo.moved_piece.kind == PieceKind::King {
            match (mv.from.to_coord().as_str(), mv.to.to_coord().as_str()) {
                ("e1", "g1") => {
                    self.set_piece_at(
                        Square::from_str("h1").expect("valid"),
                        Some(Piece {
                            color: Color::White,
                            kind: PieceKind::Rook,
                        }),
                    );
                    self.set_piece_at(Square::from_str("f1").expect("valid"), None);
                }
                ("e1", "c1") => {
                    self.set_piece_at(
                        Square::from_str("a1").expect("valid"),
                        Some(Piece {
                            color: Color::White,
                            kind: PieceKind::Rook,
                        }),
                    );
                    self.set_piece_at(Square::from_str("d1").expect("valid"), None);
                }
                ("e8", "g8") => {
                    self.set_piece_at(
                        Square::from_str("h8").expect("valid"),
                        Some(Piece {
                            color: Color::Black,
                            kind: PieceKind::Rook,
                        }),
                    );
                    self.set_piece_at(Square::from_str("f8").expect("valid"), None);
                }
                ("e8", "c8") => {
                    self.set_piece_at(
                        Square::from_str("a8").expect("valid"),
                        Some(Piece {
                            color: Color::Black,
                            kind: PieceKind::Rook,
                        }),
                    );
                    self.set_piece_at(Square::from_str("d8").expect("valid"), None);
                }
                _ => {}
            }
        }

        if let Some((capture_square, captured_piece)) = undo.captured_piece {
            self.set_piece_at(capture_square, Some(captured_piece));
        }
        debug_assert_eq!(self.position_hash, undo.previous_position_hash);
    }

    pub fn make_null_move(&mut self) {
        self.set_en_passant(None);
        self.set_side_to_move(self.side_to_move.opposite());
    }

    fn set_side_to_move(&mut self, side_to_move: Color) {
        if self.side_to_move != side_to_move {
            self.position_hash ^= side_to_move_hash();
            self.side_to_move = side_to_move;
        }
    }

    fn set_castling_rights(&mut self, castling_rights: u8) {
        if self.castling_rights != castling_rights {
            self.position_hash ^= castling_hash(self.castling_rights);
            self.castling_rights = castling_rights;
            self.position_hash ^= castling_hash(self.castling_rights);
        }
    }

    fn set_en_passant(&mut self, en_passant: Option<Square>) {
        if self.en_passant != en_passant {
            if let Some(square) = self.en_passant {
                self.position_hash ^= en_passant_hash(square);
            }
            self.en_passant = en_passant;
            if let Some(square) = self.en_passant {
                self.position_hash ^= en_passant_hash(square);
            }
        }
    }

    fn rebuild_bitboards(&mut self) {
        self.bitboards = [0; 12];
        for (index, piece) in self.squares.iter().enumerate() {
            if let Some(piece) = piece {
                self.bitboards[piece_index(*piece)] |= 1_u64 << index;
            }
        }
    }

    fn recompute_position_hash(&self) -> u64 {
        let mut hash = castling_hash(self.castling_rights);
        if self.side_to_move == Color::Black {
            hash ^= side_to_move_hash();
        }
        if let Some(square) = self.en_passant {
            hash ^= en_passant_hash(square);
        }
        for (index, piece) in self.squares.iter().enumerate() {
            if let Some(piece) = piece {
                hash ^= piece_hash(*piece, Square(index as u8));
            }
        }
        hash
    }

    pub fn game_status(&mut self) -> GameStatus {
        let legal_moves = self.generate_legal_moves();
        if legal_moves.is_empty() {
            if self.in_check(self.side_to_move) {
                GameStatus::Checkmate(self.side_to_move)
            } else {
                GameStatus::Stalemate
            }
        } else if self.halfmove_clock >= 100 {
            GameStatus::DrawByFiftyMoveRule
        } else if self.is_threefold_repetition() {
            GameStatus::DrawByRepetition
        } else {
            GameStatus::Ongoing
        }
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = self.generate_legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut nodes = 0;
        for mv in moves {
            let undo = self.make_move(mv).expect("generated move must be legal");
            nodes += self.perft(depth - 1);
            self.unmake_move(mv, undo);
        }
        nodes
    }
}

fn piece_index(piece: Piece) -> usize {
    let color_offset = match piece.color {
        Color::White => 0,
        Color::Black => 6,
    };
    let kind_offset = match piece.kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    };
    color_offset + kind_offset
}

fn for_each_bit(mut bitboard: Bitboard, mut callback: impl FnMut(Square)) {
    while bitboard != 0 {
        callback(pop_lsb(&mut bitboard));
    }
}

fn white_single_pawn_pushes(pawns: Bitboard, occupied: Bitboard) -> Bitboard {
    pawns << 8 & !occupied
}

fn black_single_pawn_pushes(pawns: Bitboard, occupied: Bitboard) -> Bitboard {
    pawns >> 8 & !occupied
}

fn append_pawn_moves(mut targets: Bitboard, from_delta: i8, moves: &mut Vec<ChessMove>) {
    while targets != 0 {
        let to = pop_lsb(&mut targets);
        let from = Square((to.0 as i16 + from_delta as i16) as u8);
        moves.push(ChessMove::new(from, to, None));
    }
}

fn append_pawn_promotions(mut targets: Bitboard, from_delta: i8, moves: &mut Vec<ChessMove>) {
    while targets != 0 {
        let to = pop_lsb(&mut targets);
        let from = Square((to.0 as i16 + from_delta as i16) as u8);
        for promotion in [
            PieceKind::Queen,
            PieceKind::Rook,
            PieceKind::Bishop,
            PieceKind::Knight,
        ] {
            moves.push(ChessMove::new(from, to, Some(promotion)));
        }
    }
}

fn pop_lsb(bitboard: &mut Bitboard) -> Square {
    let square = Square(bitboard.trailing_zeros() as u8);
    *bitboard &= *bitboard - 1;
    square
}

fn knight_attacks(square: Square) -> Bitboard {
    attack_mask(
        square,
        &[
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ],
    )
}

fn king_attacks(square: Square) -> Bitboard {
    attack_mask(
        square,
        &[
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
    )
}

fn bishop_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    sliding_attack_mask(square, occupied, &[(-1, -1), (-1, 1), (1, -1), (1, 1)])
}

fn rook_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    sliding_attack_mask(square, occupied, &[(-1, 0), (1, 0), (0, -1), (0, 1)])
}

fn queen_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(square, occupied) | rook_attacks(square, occupied)
}

fn sliding_attack_mask(square: Square, occupied: Bitboard, directions: &[(i8, i8)]) -> Bitboard {
    let mut attacks = 0;
    for &(file_delta, rank_delta) in directions {
        let mut current = square;
        while let Some(target) = current.offset(file_delta, rank_delta) {
            let target_bit = 1_u64 << target.0;
            attacks |= target_bit;
            if occupied & target_bit != 0 {
                break;
            }
            current = target;
        }
    }
    attacks
}

fn attack_mask(square: Square, deltas: &[(i8, i8)]) -> Bitboard {
    deltas.iter().fold(0, |mask, &(file_delta, rank_delta)| {
        square
            .offset(file_delta, rank_delta)
            .map_or(mask, |target| mask | (1_u64 << target.0))
    })
}

fn piece_hash(piece: Piece, square: Square) -> u64 {
    let color = match piece.color {
        Color::White => 0_u64,
        Color::Black => 1,
    };
    let kind = match piece.kind {
        PieceKind::Pawn => 0_u64,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    };
    zobrist_mix(0x9e37_79b9_7f4a_7c15 ^ ((color * 6 + kind) * 64 + u64::from(square.0)))
}

fn side_to_move_hash() -> u64 {
    zobrist_mix(0xbf58_476d_1ce4_e5b9)
}

fn castling_hash(rights: u8) -> u64 {
    zobrist_mix(0x94d0_49bb_1331_11eb ^ u64::from(rights))
}

fn en_passant_hash(square: Square) -> u64 {
    zobrist_mix(0xda94_2042_e4dd_58b5 ^ u64::from(square.0))
}

fn zobrist_mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameStatus {
    Ongoing,
    Checkmate(Color),
    Stalemate,
    DrawByRepetition,
    DrawByFiftyMoveRule,
}

pub fn piece_unicode(piece: Piece) -> char {
    match (piece.color, piece.kind) {
        (Color::White, PieceKind::Pawn) => '♙',
        (Color::White, PieceKind::Knight) => '♘',
        (Color::White, PieceKind::Bishop) => '♗',
        (Color::White, PieceKind::Rook) => '♖',
        (Color::White, PieceKind::Queen) => '♕',
        (Color::White, PieceKind::King) => '♔',
        (Color::Black, PieceKind::Pawn) => '♟',
        (Color::Black, PieceKind::Knight) => '♞',
        (Color::Black, PieceKind::Bishop) => '♝',
        (Color::Black, PieceKind::Rook) => '♜',
        (Color::Black, PieceKind::Queen) => '♛',
        (Color::Black, PieceKind::King) => '♚',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_round_trip() {
        let board = Board::from_fen(CLASSIC_STARTPOS_FEN).unwrap();
        assert_eq!(board.to_fen(), CLASSIC_STARTPOS_FEN);
    }

    #[test]
    fn parse_and_format_move() {
        let mv = ChessMove::from_uci("e7e8q").unwrap();
        assert_eq!(mv.to_uci(), "e7e8q");
    }

    #[test]
    fn startpos_perft_regression() {
        let mut board = Board::startpos();
        assert_eq!(board.perft(1), 20);
        assert_eq!(board.perft(2), 400);
        assert_eq!(board.perft(3), 8_902);
        assert_eq!(board.perft(4), 197_281);
    }

    #[test]
    fn kiwipete_perft_regression() {
        let mut board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        assert_eq!(board.perft(1), 48);
        assert_eq!(board.perft(2), 2_039);
        assert_eq!(board.perft(3), 97_862);
    }

    #[test]
    fn en_passant_is_legal_when_available() {
        let mut board =
            Board::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .unwrap();
        let legal = board.generate_legal_moves();
        assert!(legal.iter().any(|mv| mv.to_uci() == "e5d6"));
    }

    #[test]
    fn position_hash_round_trips_special_moves() {
        let cases = [
            (CLASSIC_STARTPOS_FEN, "e2e4"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ];

        for (fen, uci) in cases {
            let mut board = Board::from_fen(fen).unwrap();
            let initial_hash = board.position_hash();
            let mv = board.parse_uci_move(uci).unwrap();
            let undo = board.make_move(mv).unwrap();
            assert_ne!(board.position_hash(), initial_hash);
            board.unmake_move(mv, undo);
            assert_eq!(board.position_hash(), initial_hash);
        }
    }

    #[test]
    fn bitboards_expose_start_position_piece_sets() {
        let board = Board::startpos();
        assert_eq!(board.occupancy(Color::White).count_ones(), 16);
        assert_eq!(board.occupancy(Color::Black).count_ones(), 16);
        assert_eq!(
            board.pieces(Color::White, PieceKind::Knight),
            0x0000_0000_0000_0042
        );
        assert_eq!(
            board.pieces(Color::Black, PieceKind::Knight),
            0x4200_0000_0000_0000
        );
    }

    #[test]
    fn knight_attack_mask_has_no_file_wraparound() {
        let b1 = Square::from_str("b1").unwrap();
        assert_eq!(knight_attacks(b1), 0x0000_0000_0005_0800);
    }

    #[test]
    fn bishop_attack_mask_stops_at_the_first_blocker() {
        let d4 = Square::from_str("d4").unwrap();
        let e5 = Square::from_str("e5").unwrap();
        let f6 = Square::from_str("f6").unwrap();
        let attacks = bishop_attacks(d4, 1_u64 << e5.0);
        assert_ne!(attacks & (1_u64 << e5.0), 0);
        assert_eq!(attacks & (1_u64 << f6.0), 0);
    }

    #[test]
    fn white_pawn_push_mask_excludes_blocked_pawns() {
        let board = Board::from_fen("4k3/8/8/8/8/4p3/3PP3/4K3 w - - 0 1").unwrap();
        let occupied = board.occupancy(Color::White) | board.occupancy(Color::Black);
        assert_eq!(
            white_single_pawn_pushes(board.pieces(Color::White, PieceKind::Pawn), occupied),
            1_u64 << Square::from_str("d3").unwrap().0
        );
    }

    #[test]
    fn randomized_legal_play_preserves_fen_and_hash_invariants() {
        let mut seed = 0x9e37_79b9_u64;
        for _game in 0..12 {
            let mut board = Board::startpos();
            for _ply in 0..80 {
                let fen = board.to_fen();
                let rebuilt = Board::from_fen(&fen).unwrap();
                assert_eq!(rebuilt.to_fen(), fen);
                assert_eq!(rebuilt.position_hash(), board.position_hash());

                let moves = board.generate_legal_moves();
                if moves.is_empty() {
                    break;
                }
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mv = moves[(seed as usize) % moves.len()];
                let initial_hash = board.position_hash();
                let undo = board.make_move(mv).unwrap();
                board.unmake_move(mv, undo);
                assert_eq!(board.position_hash(), initial_hash);
                board.make_move(mv).unwrap();
            }
        }
    }
}

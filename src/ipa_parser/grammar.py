grammar_str = """?segment : vowel | consonant

//
// Vowels
//

vowel : [pre_feature*] vowel_core

?vowel_core : monophthong
    | diphthong
    | triphthong

?monophthong : vowel_atom
diphthong  : polyphthong_element polyphthong_element
triphthong  : polyphthong_element vowel_atom polyphthong_element

?polyphthong_element : onset_coda | vowel_atom
onset_coda : w_j post_feature*
?w_j : /w|ɰ/ -> w | "j" -> j
vowel_atom : vowel_glyph post_feature*
?vowel_glyph : apical_vowel | regular_vowel

?apical_vowel : "ɿ" -> alveolar_apical_vowel_unrounded
    | "ʮ"           -> alveolar_apical_vowel_rounded
    | "ʅ"           -> postalveolar_apical_vowel_unrounded
    | "ʯ"           -> postalveolar_apical_vowel_rounded

// Some special cases must be handled at the pre-processing step
// because diacritics denoting vowel height are ambiguous there.
?regular_vowel : /a(\u0308)?/ -> open_central_unrounded
    | "E1"                    -> mid_front_unrounded            // e\u031e
    | "e"                     -> close_mid_front_unrounded
    | "i"                     -> close_front_unrounded
    | "O1"                    -> mid_back_rounded               // o\u031e
    | "o"                     -> close_mid_back_rounded
    | "u"                     -> close_back_rounded
    | "y"                     -> close_front_rounded
    | "æ"                     -> near_open_front_unrounded
    | "O2"                    -> mid_front_rounded              // ø\u031e
    | "ø"                     -> close_mid_front_rounded
    | "œ"                     -> open_mid_front_rounded
    | "ɐ"                     -> near_open_central_unrounded
    | "ɑ"                     -> open_back_unrounded
    | "A1"                    -> open_central_rounded           // ɒ\u0308
    | "ɒ"                     -> open_back_rounded
    | "ɔ"                     -> open_mid_back_rounded
    | "ɘ"                     -> close_mid_central_unrounded
    | "ə"                     -> mid_central_unrounded
    | "ɛ"                     -> open_mid_front_unrounded
    | "ɜ"                     -> open_mid_central_unrounded
    | "ɞ"                     -> open_mid_central_rounded
    | "Y1"                    -> mid_back_unrounded              // ɤ\u031e
    | "ɤ"                     -> close_mid_back_unrounded
    | "I1"                    -> near_close_central_unrounded    // ɨ\u031e
    | "ɨ"                     -> close_central_unrounded
    | "ɪ"                     -> near_close_front_unrounded
    | "W1"                    -> near_close_back_unrounded       // ɯ\u031e
    | "ɯ"                     -> close_back_unrounded
    | "ɵ"                     -> close_mid_central_rounded
    | "ɶ"                     -> open_front_rounded
    | "U1"                    -> near_close_central_rounded      // ʉ\u031e
    | "ʉ"                     -> close_central_rounded
    | "ʊ"                     -> near_close_back_rounded
    | "ʌ"                     -> open_mid_back_unrounded
    | "ʏ"                     -> near_close_front_rounded


//
// Consonants
//

consonant : pre_feature* consonant_core

// special mutli-glyph cases go here
?consonant_core : simple_consonant | affricate // | click

affricate : stop fricative
    | stop trill
    | stop tap

%ignore "\u0361"  // ɠ͡ɓ
?simple_consonant: stop
    | fricative
    | approximant
    | tap
    | trill

stop        : stop_core post_feature*
fricative   : fricative_core post_feature*
approximant : approximant_core post_feature*
tap         : tap_core post_feature*
trill       : trill_core post_feature*

?stop_core : "p" -> voiceless_bilabial_plosive
    | "b" -> voiced_bilabial_plosive
    | "t" -> voiceless_alveolar_plosive
    | "d" -> voiced_alveolar_plosive
    | "ʈ" -> voiceless_retroflex_plosive
    | "ɖ" -> voiced_retroflex_plosive
    | "c" -> voiceless_palatal_plosive
    | "ɟ" -> voiced_palatal_plosive
    | "k" -> voiceless_velar_plosive
    | /ɡ|g/ -> voiced_velar_plosive
    | "q" -> voiceless_uvular_plosive
    | "ɢ" -> voiced_uvular_plosive
    | "ʡ" -> epiglottal_plosive
    | "ʔ" -> glottal_stop
    | "tp" -> voiceless_labial_alveolar_plosive
    | "db" -> voiced_labial_alveolar_plosive
    | "kp" -> voiceless_labial_velar_plosive
    | "ɡb" -> voiced_labial_velar_plosive
    | "qʡ" -> voiceless_uvular_epiglottal_plosive
    | "m" -> voiced_bilabial_nasal_plosive
    | "ɱ" -> voiced_labiodental_nasal_plosive
    | "n" -> voiced_alveolar_nasal_plosive
    | "ɳ" -> voiced_retroflex_nasal_plosive
    | "ɲ" -> voiced_palatal_nasal_plosive
    | "ŋ" -> voiced_velar_nasal_plosive
    | "ɴ" -> voiced_uvular_nasal_plosive
    | "nm" -> voiced_labial_alveolar_nasal_plosive
    | "n̊m̊" -> voiceless_labial_alveolar_nasal_plosive
    | "ŋm" -> voiced_labial_velar_nasal_plosive
    | "ŋ̊m̊" -> voiceless_labial_velar_nasal_plosive
    | "ɓ" -> voiced_bilabial_implosive
    | "ɗ" -> voiced_alveolar_implosive
    | "ᶑ" -> voiced_retroflex_implosive
    | "ʄ" -> voiced_palatal_implosive
    | "ɠ" -> voiced_velar_implosive
    | "ʛ" -> voiced_uvular_implosive
    | /ɠɓ|ɡɓ/ -> voiced_labial_velar_implosive

?fricative_core : "s" -> voiceless_alveolar_fricative
    | "z" -> voiced_alveolar_fricative
    | "S1" -> voiceless_hissing_hushing_fricative    // ŝ
    | "Z1" -> voiced_hissing_hushing_fricative       // ẑ
    | "ʃ" -> voiceless_postalveolar_fricative
    | "ʒ" -> voiced_postalveolar_fricative
    | "ɕ" -> voiceless_alveolo_palatal_fricative
    | "ʑ" -> voiced_alveolo_palatal_fricative
    | "ʂ" -> voiceless_retroflex_fricative
    | "ʐ" -> voiced_retroflex_fricative
    | "ɸ" -> voiceless_bilabial_fricative
    | "β" -> voiced_bilabial_fricative
    | "f" -> voiceless_labiodental_fricative
    | "v" -> voiced_labiodental_fricative
    | "θ" -> voiceless_interdental_fricative
    | "ð" -> voiced_interdental_fricative
    | "ç" -> voiceless_palatal_fricative
    | "ʝ" -> voiced_palatal_fricative
    | "x" -> voiceless_velar_fricative
    | "ɣ" -> voiced_velar_fricative
    | "ɧ" -> voiceless_palatal_velar_fricative
    | "χ" -> voiceless_uvular_fricative
    | "ʁ" -> voiced_uvular_fricative
    | "ħ" -> voiceless_pharyngeal_fricative
    | "ʕ" -> voiced_pharyngeal_fricative
    | "ʜ" -> voiceless_epiglottal_fricative
    | "ʢ" -> voiced_epiglottal_fricative
    | "ɬ" -> voiceless_alveolar_lateral_fricative
    | "ɮ" -> voiced_alveolar_lateral_fricative
    | /ꞎ|(L1)/ -> voiceless_retroflex_lateral_fricative  // ɭ̝̊
    | "L11" -> voiced_retroflex_lateral_fricative        // ɭ̝
    | "L2" -> voiceless_palatal_lateral_fricative        // ʎ̝̊
    | "L22" -> voiced_palatal_lateral_fricative          // ʎ̝
    | "L3" -> voiceless_velar_lateral_fricative          // ʟ̝̊
    | "L4" -> voiced_velar_lateral_fricative             // ʟ̝
    | "h" -> voiceless_glottal_fricative
    | "ɦ" -> voiced_glottal_fricative

?approximant_core : "D1" -> voiceless_interdental_approximant    // θ̞
    | "D2" -> voiced_interdental_approximant                     // ð̞
    | "V1" -> voiceless_bilabial_approximant                     // ɸ̞
    | "V2" -> voiced_bilabial_approximant                        // β̞
    | "R1" -> voiced_uvular_approximant                          // ʁ̞
    | "ʋ" -> voiced_labiodental_approximant
    | "ɹ" -> voiced_alveolar_approximant
    | "ɻ" -> voiced_retroflex_approximant
    | "j" -> voiced_palatal_approximant
    | "ɰ" -> voiced_velar_approximant
    | "l" -> voiced_alveolar_lateral_approximant
    | "ɭ" -> voiced_retroflex_lateral_approximant
    | "ʎ" -> voiced_palatal_lateral_approximant
    | "ʟ" -> voiced_velar_lateral_approximant
    | "L5" -> voiced_uvular_lateral_approximant                   // ʟ̠
    | "ʍ" -> voiceless_labio_velar_approximant
    | "w" -> voiced_labio_velar_approximant
    | "ɥ" -> voiced_labio_palatal_approximant

?tap_core : /ɾ|ᴅ/ -> voiced_alveolar_tap
    | "ɺ" -> voiced_alveolar_lateral_tap
    | "ɽ" -> voiced_retroflex_tap
    | "ⱱ" -> voiced_labiodental_tap  // NB: some fonts represent this character
                                     // incorrectly, but not to worry.

?trill_core : "r" -> voiced_alveolar_trill
    | "ʙ" -> voiced_bilabial_trill
    | "ʀ" -> voiced_uvular_trill

//
// Additional articulations
//

?pre_feature : /[ʰʱ]/   -> pre_aspirated
    | /[ˀʼ]/            -> pre_glottalised
    | /[ⁿnmŋɳɲɱɴ]/      -> pre_nasalised
    | "ʷ"               -> pre_labialised

?post_feature : "="         -> overlong
    | /[ː:]/                -> long
    | "ˑ"                   -> half_long
    | "ˤ"                   -> pharyngealised
    | "\u0303"              -> nasalised       // ã
    | "\u0306"              -> shortened       // ă
    | "\u031d"              -> raised          // a̝
    | "\u031e"              -> lowered         // a̞
    | "\u031f"              -> advanced        // a̟
    | "\u0320"              -> retracted       // a̠
    | "\u0324"              -> breathy_voiced  // a̤
    | /(\u030a)|(\u0325)/   -> voiceless       // å | ḁ
    | "\u0330"              -> creaky_voiced   // a̰
    | "\u2193"              -> ingressive      // a↓
    | "\u02de"              -> rhotacised      // a˞
    | "\u0348"              -> strong_articulation  // a͈
    | "\u0308"              -> centralised      // ä
    | "\u0318"              -> atr              // a̘
    | "\u0319"              -> rtr              // a̙
    | "\u031c"              -> less_rounded     // a̜
    | "\u0339"              -> more_rounded     // a̹
    | "\u032f"              -> non_syllabic     // a̯
    | "\u033d"              -> mid_centralised  //  a̽
    | /[ʰʱ]/                -> aspirated
    | "ʲ"                   -> palatalised
    | "ʷ"                   -> labialised
    | /[’ʼ']/               -> ejective
    | "ˀ"                   -> glottalised       // d
    | /ˠ|\u0334/            -> velarised         // l̴
    | "\u02e1"              -> lateral_released  // dˡ
    | "\u031a"              -> unreleased        // a̚
    | "\u0329"              -> syllabic          // a̩
    | "\u032a"              -> dental            // a̪
    | "\u0347"              -> alveolar          // a͇
    | "\u033a"              -> apical            // a̺
    | "\u033b"              -> laminal           // a̻
    | "\u0349"              -> weakly_articulated  // a͉
    | "\u1da3"              -> labio_palatalised   // aᶣ
    | "ⁿ"                   -> nasal_released
    | /[ᶻˢ]/                -> affricated
    | "ᴱ"                   -> epilaryngeal_source
    | "\u0353"              -> frictionalised  // a͓
    | "\u02ed"              -> tenuis          // t˭
    | "\u032c"              -> voiced
    | "\u033c"              -> linguo_labial

%import common.WS
%ignore WS"""
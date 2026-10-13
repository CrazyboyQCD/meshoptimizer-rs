//! Marching-cubes triangulation tables for the remesher. Upstream: the case
//! table and `buildRemeshTables` rotation propagation (remesher.cpp).
//!
//! Upstream builds these once via C++ static initialization; the port rebuilds
//! them per call (pure function of the base case table, behavior-identical).

// triangles are encoded as three vertex codes in three nibbles
// vertex codes match the bit order XYZ (e.g. 3 maps to X=1 Y=1 Z=0)
struct Case {
    code: u8,
    altmode: u8,
    triangles: [u16; 5], // up to 4 triangles and a null terminator
    alternate: [u16; 5], // alternate triangulation, restricted to have <= triangles vs primary
}

// the base cases establish a canonical triangulation; unlike classical Marching Cubes, we connect cube *corners* instead of edges
// each case may have an alternate triangulation, used based on deciders; crucially, open boundary edges must stay consistent as decider varies per cell
const BASE_CASES: [Case; 24] = [
    Case {
        code: 0x00,
        altmode: 0,
        triangles: [0; 5],
        alternate: [0; 5],
    },
    Case {
        code: 0x01,
        altmode: 0,
        triangles: [0; 5],
        alternate: [0; 5],
    },
    Case {
        code: 0x03,
        altmode: 0,
        triangles: [0; 5],
        alternate: [0; 5],
    },
    Case {
        code: 0x06,
        altmode: 0,
        triangles: [0; 5],
        alternate: [0; 5],
    },
    Case {
        code: 0x07,
        altmode: 0,
        triangles: [0x120, 0, 0, 0, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x0f,
        altmode: 2,
        triangles: [0x130, 0x032, 0, 0, 0],
        alternate: [0x321, 0x120, 0, 0, 0],
    },
    Case {
        code: 0x16,
        altmode: 1,
        triangles: [0x124, 0x142, 0, 0, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x17,
        altmode: 0,
        triangles: [0x124, 0, 0, 0, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x18,
        altmode: 0,
        triangles: [0; 5],
        alternate: [0; 5],
    },
    Case {
        code: 0x19,
        altmode: 1,
        triangles: [0x043, 0x403, 0, 0, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x1b,
        altmode: 1,
        triangles: [0x134, 0x430, 0, 0, 0],
        alternate: [0x410, 0x013, 0, 0, 0],
    },
    Case {
        code: 0x1d,
        altmode: 1,
        triangles: [0x243, 0x340, 0, 0, 0],
        alternate: [0x320, 0x024, 0, 0, 0],
    },
    Case {
        code: 0x1e,
        altmode: 1,
        triangles: [0x142, 0x432, 0x341, 0, 0],
        alternate: [0x132, 0, 0, 0, 0],
    },
    Case {
        code: 0x1f,
        altmode: 2,
        triangles: [0x412, 0x213, 0, 0, 0],
        alternate: [0x134, 0x432, 0, 0, 0],
    },
    Case {
        code: 0x3c,
        altmode: 1,
        triangles: [0x345, 0x324, 0x543, 0x423, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x3d,
        altmode: 1,
        triangles: [0x253, 0x524, 0x350, 0, 0],
        alternate: [0x240, 0x504, 0x320, 0, 0],
    },
    Case {
        code: 0x3f,
        altmode: 2,
        triangles: [0x325, 0x524, 0, 0, 0],
        alternate: [0x243, 0x345, 0, 0, 0],
    },
    Case {
        code: 0x69,
        altmode: 1,
        triangles: [0x035, 0x560, 0x306, 0x536, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x6b,
        altmode: 1,
        triangles: [0x063, 0x605, 0x365, 0, 0],
        alternate: [0x305, 0, 0, 0, 0],
    },
    Case {
        code: 0x6f,
        altmode: 1,
        triangles: [0x365, 0x560, 0, 0, 0],
        alternate: [0x530, 0x036, 0, 0, 0],
    },
    Case {
        code: 0x7e,
        altmode: 0,
        triangles: [0x142, 0x536, 0, 0, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0x7f,
        altmode: 0,
        triangles: [0x365, 0, 0, 0, 0],
        alternate: [0; 5],
    },
    Case {
        code: 0xf0,
        altmode: 2,
        triangles: [0x547, 0x746, 0, 0, 0],
        alternate: [0x465, 0x567, 0, 0, 0],
    },
    Case {
        code: 0xff,
        altmode: 0,
        triangles: [0; 5],
        alternate: [0; 5],
    },
];

pub(super) struct RemeshTables {
    pub(super) triangles: [[[u16; 5]; 2]; 256],
    pub(super) count: [u8; 256],
    pub(super) alt: [u8; 256],
}

// for each cube case, a triangulation is derived from the base cases via 90 degree rotations
pub(super) fn build_remesh_tables() -> RemeshTables {
    // corner remap for 90 degree rotations around X and Y
    const ROTATIONS: [[u8; 8]; 2] = [[2, 3, 6, 7, 0, 1, 4, 5], [4, 0, 6, 2, 5, 1, 7, 3]];

    let mut tables = RemeshTables {
        triangles: [[[0; 5]; 2]; 256],
        count: [0; 256],
        alt: [0; 256],
    };
    let mut filled = [0u8; 256];

    // copy base cases as is; we will generate the rest via rotations
    for case in &BASE_CASES {
        tables.triangles[case.code as usize][0] = case.triangles;
        tables.triangles[case.code as usize][1] = if case.altmode != 0 {
            case.alternate
        } else {
            case.triangles
        };
        tables.alt[case.code as usize] = case.altmode;
        filled[case.code as usize] = 1;
    }

    // propagate base cases iteratively to fill all cube cases via rotations
    // note: the worst case is 5 passes but in practice we only need 3 to fill all cases
    for _ in 0..3 {
        for code in 0..256usize {
            if filled[code] != 1 {
                continue;
            }

            for rotations in &ROTATIONS {
                let mut rotated = 0u8;
                for (i, &rot) in rotations.iter().enumerate() {
                    rotated |= (((code >> i) & 1) << rot) as u8;
                }

                let rotated = rotated as usize;
                if filled[rotated] != 0 {
                    continue;
                }

                for k in 0..2 {
                    for i in 0..5 {
                        let tri = tables.triangles[code][k][i];
                        if tri == 0 {
                            break;
                        }

                        let mapped = (u16::from(rotations[((tri >> 8) & 0xf) as usize]) << 8)
                            | (u16::from(rotations[((tri >> 4) & 0xf) as usize]) << 4)
                            | u16::from(rotations[(tri & 0xf) as usize]);
                        tables.triangles[rotated][k][i] = mapped;
                    }
                }

                tables.alt[rotated] = tables.alt[code];
                filled[rotated] = 1; // mark as pending
            }

            filled[code] = 2; // mark as processed
        }
    }

    // finalize auxiliary tables
    for code in 0..256usize {
        let mut count = 0usize;
        while tables.triangles[code][0][count] != 0 {
            count += 1;
        }

        tables.count[code] = count as u8;

        // counting pass does not use alternate triangulations, so they need to stay
        // at or below primary triangle count (upstream asserts this invariant)
    }

    tables
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: none (internal marching-cubes tables;
    // behavior pinned through the meshopt_remesh fixtures and the invariants
    // asserted here).
    use super::*;

    #[test]
    fn remesh_tables_cover_all_codes_and_terminate() {
        let tables = build_remesh_tables();

        for code in 0..256usize {
            let count = tables.count[code] as usize;

            // every code is filled via rotations and carries a null terminator
            assert!(count <= 4, "code {code:#x} triangle count {count}");
            assert_eq!(
                tables.triangles[code][0][count], 0,
                "code {code:#x} primary missing terminator"
            );
            assert_eq!(
                tables.triangles[code][1][count], 0,
                "code {code:#x} alternate missing terminator"
            );

            // base cases keep their configured mode
            if let Some(case) = BASE_CASES.iter().find(|c| c.code as usize == code) {
                assert_eq!(tables.alt[code], case.altmode, "code {code:#x} altmode");
            }
        }

        assert_eq!(tables.count[0x07], 1);
        assert_eq!(tables.alt[0x0f], 2);
        assert_eq!(tables.count[0xff], 0);
    }
}

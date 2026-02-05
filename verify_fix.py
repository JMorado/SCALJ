import numpy as np

from scalj.dataset import create_entries_from_ml_output


def verify():
    mixture_id = "test"
    smiles = "C"
    # Create fake data: 3 frames, 5 atoms
    coords = [np.random.rand(5, 3) for _ in range(3)]
    box_vectors = [np.eye(3) for _ in range(3)]
    energies = np.array([1.0, 2.0, 3.0])
    forces = np.random.rand(3, 5, 3)  # 3 frames, 5 atoms, 3 dims

    entries = create_entries_from_ml_output(
        mixture_id, smiles, coords, box_vectors, energies, forces
    )

    print(f"Number of entries: {len(entries)}")
    if len(entries) == 1:
        e = entries[0]
        print(f"Energy length: {len(e['energy'])}")
        print(f"Coords length: {len(e['coords'])}")
        # Check lengths
        # Energy should have 3 values
        # Coords should have 3 * 5 * 3 = 45 values
        is_energy_correct = len(e["energy"]) == 3
        is_coords_correct = len(e["coords"]) == 3 * 5 * 3

        if is_energy_correct and is_coords_correct:
            print("SUCCESS: Aggregation correct.")
        else:
            print(
                f"FAILURE: Incorrect dimensions. Energy: {len(e['energy'])}, Coords: {len(e['coords'])}"
            )
    else:
        print(f"FAILURE: Multiple entries returned ({len(entries)}).")


if __name__ == "__main__":
    verify()

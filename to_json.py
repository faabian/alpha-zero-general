import json
import fire
import numpy as np
from pathlib import Path
import pickle

from othello.OthelloGame import OthelloGame


def save_jsonl(checkpoint_dir: Path, output_path: Path):
    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    examples = []
    for fn in checkpoint_dir.glob("*examples"):
        with open(fn, "rb") as f:
            read_all = pickle.load(f)
            assert len(read_all) == 1
            read = read_all[-1]
            print(f"Read {len(read)} examples from {fn}")
            examples += read

    print(f"Read {len(examples)} in total.")
    n = examples[0][0].shape[1]
    game = OthelloGame(n)

    def example_to_dict(example):
        """
        Encode an example as in:
        `X X X X X O - X O X X X O O X O X X X X X X X X - - X X X - - - X X X X action5,4`

        Note: requires SentencePiece to split the last part as ['X', 'action', '5', ',', '4']
        as /large_experiments/fair_llm/datasets/tokenizers/tokenizer_final_32k.minus_inf_ws.model does.
        """
        board_str = game.to_text(example[0])
        action = int(np.argmax(example[1]))
        action_x, action_y = action // n, action % n
        action_str = ",".join((str(action_x), str(action_y)))
        text = board_str + "action" + action_str
        return {
            "text": text,
            "board": example[0].tolist(),
            "action_id": action,
            "action": (action_x, action_y),
            "result": example[2],
        }

    assert output_path.name.endswith(".jsonl"), output_path
    with open(output_path, "w") as f:
        f.write("\n".join(json.dumps(example_to_dict(ex)) for ex in examples) + "\n")


if __name__ == "__main__":
    fire.Fire(save_jsonl)

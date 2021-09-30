import pathlib
import argparse
import numpy as np
import xgboost as xgb
import soundfile as sf
from typing import Union


def parse_args():
    parser = argparse.ArgumentParser(description="Predict xgb model")
    parser.add_argument(
        "--path_file",
        type=str,
        required=True,
        default=None,
        help="Path to sound file .wav",
    )
    parser.add_argument(
        "--path_model",
        type=str,
        required=True,
        default=None,
        help="Path to load pretrain xgb model",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=38016,
        help="Set length"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Set target"
    )
    args = parser.parse_args()
    return args


def test_gender(
    path_file: Union[str, pathlib.Path],
    path_model: str,
    length: int = 38016,
    target: int = None,
) -> None:
    """
    path_file : str, path to wave file
    path_model: str, path to pretrain model(xgb)
    length: int, len wave
    target: None, for test and may by to equal if you have label
    Make predict for pretrain model(xgb)
    """
    model = xgb.XGBClassifier()
    model.load_model(path_model)

    if not isinstance(path_file, pathlib.PosixPath):
        path_file = pathlib.Path(path_file)

    d, _ = sf.read(path_file)
    if d.shape[0] < length:
        d = np.append(d, [0]*(length-d.shape[0]), axis=0)
    else:
        d = d[:length]
    d_f = np.fft.fft(d)[:len(d)//2]
    d_f = d_f.reshape(352, 54).mean(axis=1)
    y_ = model.predict_proba(d_f.astype('float64').reshape((1, -1)))
    idx = np.argmax(y_)
    if idx == 1:gender = 'Man'
    else: gender = 'Woman'
    result_str = f'Predict id: {path_file.stem}, score: {y_[0][idx]}, prob: {y_[0]},  gender: {gender}'
    ans = False
    if target == 0 or target == 1:
        if idx == target:
            ans = True
            result_str = result_str + f' Target: {ans}'
        else:
            result_str = result_str + f' Target: {ans}'
    print(result_str)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    args = parse_args()
    test_gender(
        args.path_file,
        args.path_model,
        args.length,
        args.target
    )

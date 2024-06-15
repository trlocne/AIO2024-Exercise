import math
import numpy as np

# 1. Viết function thực hiện đánh giá classification model bằng F1-Score.
def compute_f1_score(tp, fp, fn):
    check = False
    if type(tp) is not int:
        check = True
        print("tp must be int")
    if type(fp) is not int:
        check = True
        print("fp must be int")
    if type(fn) is not int:
        check = True
        print("fn must be int")

    if check:
        return False

    if tp < 0 or fp < 0 or fn < 0:
        print("fn and fp and fn must be greater than zero")
        return False

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print(f"precision is {round(precision, 2)}")
    print(f"recall is {round(recall, 2)}")
    print(f"f1_score is {round(f1_score, 2)}")
    return True

# 2. Viết function mô tả theo 3 activation function.
def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def simulate_activate_func():
    x = input("Mời bạn nhập giá trị x ")
    act_func = input(
        "Mời bạn nhận tên activation function chỉ có 3 loại (sigmoid, relu, elu) ")
    if is_number(x):
        x = float(x)
        if act_func == 'sigmoid':
            result = 1 / (1 + math.e**(-x))
            print(f"{act_func}: f({x}) = {result}")
        elif act_func == 'relu':
            result = 0 if x <= 0 else x
            print(f"{act_func}: f({x}) = {result}")
        elif act_func == 'elu':
            result = 0.01 * (math.e**(x) - 1.0) if x <= 0 else x
            print(f"{act_func}: f({x}) = {result}")
        else:
            print(f"{act_func} is not supported")
    else:
        print("x must be a number")

# 3. Viết function lựa chọn regression loss function để tính loss:
def calc_regres_loss():
    number_samples = input("Nhập vào một number samples ")
    if not (number_samples.isnumeric()):
        print("number of samples must be an integer number")
    else:
        loss = input("Nhập vào loss name ")
        predict = []
        target = []
        rng = np.random.default_rng(0)
        for _ in range(int(number_samples)):
            predict.append(rng.uniform(0, 10))
            target.append(rng.uniform(0, 10))
        mae = 0.0
        mse = 0.0
        rmse = 0.0
        for i in range(int(number_samples)):
            los = 0.0
            if loss.upper() == "MAE":
                mae = math.sqrt(target[i] - predict[i])
                los = mae
            else:
                mse += (target[i] - predict[i])**2
                los = mse
            print(f"loss name: {loss.upper()}, sample: {i}, pred: {predict[i]}, target: {target[i]}, loss: {los}")
        mae = mae / float(number_samples)
        mse = mse / float(number_samples)
        rmse = math.sqrt(mse)
        if loss.upper() == "MAE":
            print(mae)
        elif loss.upper() == "MSE":
            print(mse)
        else:
            print(rmse)

# 4. Viết 4 function để ước lượng hàm số sau:
def appox_sin(x, n):
    sin = 0.0
    for i in range(n):
        sin += ((-1)**i) * ((x**(2*i+1)) / math.factorial((2*i+1)))
    return sin


def appox_cos(x, n):
    cos = 0.0
    for i in range(n):
        cos += ((-1)**i) * ((x**(2*i)) / math.factorial((2*i)))
    return cos


def appox_sinh(x, n):
    sinh = 0.0
    for i in range(n):
        sinh += ((x**(2*i + 1)) / math.factorial((2*i + 1)))
    return sinh

def appox_cosh(x, n):
    cosh = 0.0
    for i in range(n):
        cosh += ((x**(2*i)) / math.factorial((2*i)))
    return cosh

# 5. Viết hàm thực hiện Mean Different of nth Root Error.
def md_nre_single_sample(y, y_hat, n, p):
    return (math.pow(y, 1/n) - math.pow(y_hat, 1/n))**p

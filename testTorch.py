import torch

print(f"파이토치의 버전:{torch.__version__}")


def func_print(msg, x):
    print(msg + ":")
    print(x)
    print("==============================")


def func_notInitializedTensor():
    msg = "초기화되지 않은 텐서"
    x = torch.empty(4, 2)
    func_print(msg, x)


def func_randInitializedTensor():
    msg = "랜덤으로 초기화 된 텐서"
    x = torch.rand(4, 2)
    func_print(msg, x)


def func_zeroFilledTypeLong():
    msg = "데이터 타입이 long이고 0으로 채워진 텐서"
    x = torch.zeros(4, 2, dtype=torch.long)
    func_print(msg, x)


def func_makeTensorByInput():
    msg = "사용자가 입력한 값으로 텐서 초기화"
    x = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
    )
    func_print(msg, x)


def func_makeTensorByOne():
    msg = "2 x 4 크기, double 타입, 1로 채워진 텐서"
    x = torch.Tensor()
    x = x.new_ones(2, 4, dtype=torch.double)
    func_print(msg, x)

    """
    Q. torch.Tensor() 와 torch.tensor()의 차이점은 무엇인가요?
    PyTorch에서 torch.Tensor()와 torch.tensor() 함수는 모두 텐서를 생성하는 함수입니다.
    하지만 이 두 함수는 약간의 차이가 있습니다.
    torch.Tensor() 함수는 주어진 크기의 무작위 값으로 초기화된 텐서를 생성합니다.
    이 함수는 데이터 타입과 디바이스를 지정하지 않으면 기본적으로 float32 타입과 CPU를 사용합니다.
    반면에 torch.tensor() 함수는 주어진 데이터로부터 새로운 텐서를 생성합니다.
    데이터를 입력으로 받으므로, torch.Tensor() 함수보다는 다양한 타입과 디바이스에서 생성하기 쉽고, 특정한 값을 초기화할 수 있습니다.
    또한 이 함수는 입력 데이터가 이미 PyTorch 텐서인 경우, 새로운 메모리를 할당하는 대신 같은 메모리를 공유하는 새로운 뷰를 만듭니다.
    따라서 torch.tensor() 함수는 메모리를 더 효율적으로 사용할 수 있습니다.
    즉, torch.Tensor() 함수는 기본값으로 초기화된 무작위 값을 가지는 텐서를 생성하며, torch.tensor() 함수는 주어진 데이터로부터 새로운 텐서를 생성합니다.
    """


def func_makeRandTensor():
    msg = "생성한 텐서와 같은 크기, float타입, 무작위로 채워진 텐서 (이어서 원형텐서 비교)"
    xOrigin = torch.rand(4, 2, dtype=torch.float32)
    x = torch.randn_like(xOrigin)
    func_print(msg, x)
    msg = "  ⎣__ 원형텐서:"
    func_print(msg, xOrigin)


def func_tensorSize():
    msg = "텐서의 크기 계산"
    x = torch.rand(4, 2)
    func_print(msg, x)


def func_dataType():
    msg = "생성한 데이터타입별 텐서"

    ten_ft = torch.FloatTensor([1, 2, 3])
    print(msg)
    print(ten_ft)
    print(ten_ft.dtype)
    # 타입캐스팅
    ten_half = ten_ft.half()
    ten_short = ten_ft.short()
    ten_int = ten_ft.int()
    ten_long = ten_ft.long()
    print(ten_half.dtype)
    print(ten_short.dtype)
    print(ten_int.dtype)
    print(ten_long.dtype)

    '''
    --- 만약 GPU 를 사용시 torch.cuda.<dtype> 으로 사용 ---
    torch.FloatTensor   | 32비트 플롯 (float32, float)
    torch.DoubleTensor  | 64비트 플롯 (float64, double)
    torch.HalfTensor    | 16비트 플롯 (float16, half)
    torch.ByteTensor    | 8비트 인트(unsigned) (uint8) 
    torch.CharTensor    | 8비트 인트(signed) (int8)
    torch.ShortTensor   | 16비트 인트(int16, short)
    torch.IntTensor     | 32비트 인트(int32, int)
    torch.LongTensor    | 64비트 인트(int64, long)
    '''


def func_changeProcessor():
    print("to 메소드를 사용하여 텐서를 cpu | gpu로 옮김")
    x = torch.randn(1)
    print(x)
    print(x.item())
    print(x.dtype)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')
    # device = torch.device('cpu')
    print(device)
    # y = torch.ones_like(x, device=device)
    # print(y)
    # x = x.to('cuda')
    # print(x)


func_notInitializedTensor()
func_randInitializedTensor()
func_zeroFilledTypeLong()
func_makeTensorByInput()
func_makeTensorByOne()
func_makeRandTensor()
func_dataType()
func_changeProcessor()

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ResNet-CIFAR100')
    # your code
    # parser.add_argument('--dataset', type=str, default='CIFAR100', help="This is dataset; MNIST, CIFAR10, CIFAR100")
    parser.add_argument('--batch_size', type=int, default=64, help="")
    parser.add_argument('--learning_rate', type=float, default='0.1', help="")
    parser.add_argument('--num_epoch', type=int, default=10, help="")
    parser.add_argument('--load_model', type=int, default=0, help="")
    parser.add_argument('--model', type=str, default='resnet18', help="")
    args = parser.parse_args()
    batch_size=args.batch_size
    learning_rate= args.learning_rate
    num_epoch=args.num_epoch
    model = args.model
    print(f"{model},{args},{batch_size},{learning_rate},{num_epoch}")
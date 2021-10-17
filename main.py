import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from squeezer.containers import Batch, ModelOutput
from squeezer.distiller import Distiller
from squeezer.policy import DistillationPolicy


class Teacher(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        outputs = self.network(inputs)
        return ModelOutput(logits=outputs)


class Student(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        outputs = self.network(inputs)
        return ModelOutput(logits=outputs)


def get_loader(length: int = 10000, num_features: int = 20, num_classes: int = 4, batch_size: int = 64):
    data_tensor = torch.randn(length, num_features)
    target_tensor = torch.randint(high=num_classes, size=(length,))
    dataset = TensorDataset(data_tensor, target_tensor)

    def collate(batch: list):
        data, target = zip(*batch)
        return Batch(
            data=torch.stack(data),
            target=torch.stack(target)
        )

    return DataLoader(dataset, collate_fn=collate, batch_size=batch_size)


def main():
    torch.random.manual_seed(0xDEAD)
    input_size = 32
    num_classes = 4

    train_loader = get_loader(num_features=input_size, num_classes=num_classes)
    teacher = Teacher(input_size, num_classes, hidden_size=10)
    student = Student(input_size, num_classes)
    policy = DistillationPolicy(temperature=0.2, alpha=1.)

    distiller = Distiller(teacher, student, policy)
    distiller(train_loader, n_epochs=10)


if __name__ == '__main__':
    main()

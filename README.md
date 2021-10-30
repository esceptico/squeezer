# Squeezer (WIP)

## Usage
### Step 1: Define Distiller class
Implement `teacher_forward`, `student_forward` 
and (if required) `move_batch_to_device` methods.
```python
from squeezer.distiller import Distiller


class CustomDistiller(Distiller):
    def teacher_forward(self, batch):
        return self.teacher(batch['data'])

    def student_forward(self, batch):
        return self.student(batch['data'])
```
### Step 2: Define LossPolicy
```python
from torch.nn.functional import mse_loss

from squeezer.policy import AbstractDistillationPolicy


class DistillationPolicy(AbstractDistillationPolicy):
    def forward(self, teacher_output, student_output, batch, epoch):
        loss_mse = mse_loss(student_output, teacher_output)
        loss_dict = {'mse': loss_mse.item()}
        return loss_mse, loss_dict
```

### Step 3: Fit
```python
from torch import optim


train_loader = ...

teacher = Teacher()
student = Student()

optimizer = optim.AdamW(student.parameters(), lr=3e-4)
policy = DistillationPolicy()
distiller = CustomDistiller(teacher, student, policy, optimizer=optimizer)

distiller(train_loader, n_epochs=10)
distiller.save('path_to_some_directory')
```
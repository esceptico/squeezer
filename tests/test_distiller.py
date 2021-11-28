from torch.optim import Adam

from squeezer.distiller import Distiller


def test_distillation(loader, model, policy):
    teacher, student = model, model
    optimizer = Adam(student.parameters())

    class CustomDistiller(Distiller):
        def teacher_forward(self, batch):
            return self.teacher(batch[0])

        def student_forward(self, batch):
            return self.student(batch[0])

    distiller = CustomDistiller(teacher, student, policy, optimizer)
    distiller(loader)

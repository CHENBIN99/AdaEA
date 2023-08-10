
class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_clean = 0
        self.correct_adv = 0
        self.total_correct_clean = 0
        self.total_correct_adv = 0
        self.total_num = 0
        self.clean_acc = 0
        self.adv_acc = 0
        self.attack_rate = 0

    def update(self, correct_clean, correct_adv, total_num=1):
        self.correct_clean = correct_clean
        self.correct_adv = correct_adv
        self.total_correct_clean += correct_clean
        self.total_correct_adv += correct_adv
        self.total_num += total_num
        self.clean_acc = self.total_correct_clean / self.total_num
        self.adv_acc = self.total_correct_adv / self.total_num
        if self.clean_acc != 0:
            self.attack_rate = 1 - self.adv_acc / self.clean_acc
        else:
            self.attack_rate = 1 - self.adv_acc / (self.clean_acc + 1e-12)


class SimilarityMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.similarity = 0
        self.total_simillarity = 0
        self.total_num = 0
        self.avg_sim = 0

    def update(self, similarity, total_num=1):
        self.similarity = similarity
        self.total_simillarity += similarity
        self.total_num += total_num
        self.avg_sim = self.total_simillarity / self.total_num
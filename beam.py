
class Beam(object):
    """Ordered beam of candidate outputs."""
    
    def __init__(self, size, init_symbols, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.cuda = cuda
        self.scores = torch.FloatTensor(size).zero_()
        if self.cuda:
            self.scores = self.scores.cuda()
        
        # previous pointer
        self.prevKs = []
        
        # next step
        self.nextYs = [init_symbols]
   

    def get_current_state(self):
        return self.nextYs[-1]

    def get_current_origin(self):
        return self.prevKs[-1]

    """ Advance the beam """
    def advance(self, workd_lk):
        num_words = workd_lk.size(2)
        
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]
        
        flat_beam_lk = beam_lk.view(-1) 
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores  
    
        prev_k = bestScoresId / num_words 
        self.prevKs.append(prev_k) #
        self.nextYs.append(bestScoresId - prev_k * num_words) 
        
        # terminal 
        if self.nextYs[-1][0] == EOS_TOKEN:
            self.done = True
        return self.done 
 
    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k):
        hyp = []
        for j in range(len(self.prevKs)-1, -1, -1):
            hyp.append(self.nextYs[j + 1][k].detach().item())
            k = self.prevKs[j][k]
        return hyp[::-1]
    
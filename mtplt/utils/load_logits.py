import os
import json
import numpy as np
from collections import OrderedDict


def get_op_names(with_m=False):
    names = [
        'autocontrast_dna',
        'brightness_magnitude0.3', 'brightness_magnitude0.6', 'brightness_magnitude0.9',
        'color_magnitude0.3', 'color_magnitude0.6', 'color_magnitude0.9',
        'contrast_magnitude0.3', 'contrast_magnitude0.6', 'contrast_magnitude0.9',
        'equalize_dna',
        'invert_dna',
        'posterize_magnitude4', 'posterize_magnitude5', 'posterize_magnitude7',
        'rotate_magnitude10', 'rotate_magnitude20', 'rotate_magnitude30',
        'sharpness_magnitude0.3', 'sharpness_magnitude0.6', 'sharpness_magnitude0.9',
        'shearX_magnitude0.1', 'shearX_magnitude0.2', 'shearX_magnitude0.3',
        'shearY_magnitude0.1', 'shearY_magnitude0.2', 'shearY_magnitude0.3',
        'solarize_magnitude0', 'solarize_magnitude171', 'solarize_magnitude85.3',
        'translateX_magnitude0.151', 'translateX_magnitude0.302', 'translateX_magnitude0.453',
        'translateY_magnitude0.151', 'translateY_magnitude0.302', 'translateY_magnitude0.453',
    ]
    return names if with_m else sorted(list(set([s.split('_')[0] for s in names])))


def load_probs_dict(folder, num=11, length=-1, lexi=False):
    path = '/Users/tiankeyu/Downloads/'
    dir = os.path.join(path, folder)
    for root, dirs, fnames in os.walk(dir):
        break
    
    mops = get_op_names(True)
    mlogits, logits = {}, {}
    for fname in fnames:
        for mop in mops:
            if fname.find(mop) != -1:
                op = mop.split('_')[0]
                with open(os.path.join(dir, fname), 'r') as f:
                    li = json.load(f)
                if length == -1:
                    length = len(li)
                
                mlogits[mop] = np.array([x[2] for x in li])[np.linspace(0, length, num, dtype=int)]
                if op in logits.keys():
                    logits[op] += mlogits[mop]
                else:
                    logits[op] = mlogits[mop].copy()
    
    def post_process(ld: dict):
        sum_l = None
        for x in ld.values():
            if sum_l is None:
                sum_l = x.copy()
            else:
                sum_l += x
        for x in ld.values():
            x /= sum_l
        
        probs = OrderedDict()
        if lexi:
            for n in sorted(ld.keys(), reverse=True):
                probs[n] = ld[n]
        else:
            for _, n in sorted([(p[-1], n) for n, p in ld.items()], reverse=True):
                probs[n] = ld[n]
            
        return probs
    
    return post_process(mlogits), post_process(logits)
        

if __name__ == '__main__':
    m_probs, probs = load_probs_dict(folder='cfbest')
    print(m_probs)
    print(probs)

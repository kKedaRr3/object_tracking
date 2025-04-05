def initialize_rule_base(spatio_temporal_granules, rgb_d_granules, threshold):
    rule_base = []

    for granule in spatio_temporal_granules:
        for color_granule in rgb_d_granules:
            rule = classify_granule(granule, color_granule, threshold)
            rule_base.append(rule)

    return rule_base

#TODO depth to bedzie srednia z RGB to znaczy (R + G + B) / 3
def classify_granule(spatio_temporal_granule, rgb_d_granule, spt_threshold, c_threshold):
    decision = ""
    spatio_temporal_similarity = check_spatio_temporal_similarity(spatio_temporal_granule, spt_threshold)
    color_similarity = check_color_similarity(rgb_d_granule, c_threshold)



def check_spatio_temporal_similarity(spatio_temporal_granule, spt_threshold):
    pass

def check_color_similarity(rgb_d_granule, c_threshold):
    pass


def add_rule(self, sp_tmp, rgb, depth, decision):
    new_rule = {
        "sp_tmp": sp_tmp,
        "rgb": rgb,
        "depth": depth,
        "decision": decision
    }
    self.rules.append(new_rule)


def update_rules(self, new_info):
    """
    Gdy flow graph pokazuje duże zmiany, możemy
    zaktualizować pewne reguły na podstawie new_info.
    Tutaj kod jest symboliczny.
    """
    pass

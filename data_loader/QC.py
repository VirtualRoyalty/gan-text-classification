QC_LABEL_LIST = ["UNK_UNK", "ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc",
              "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body",
              "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed",
              "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang",
              "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product",
              "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol",
              "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc",
              "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country",
              "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count",
              "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other",
              "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize",
              "NUM_weight"]


def get_qc_dataset(input_file):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(input_file, 'r') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list[1:]:
            split = line.split(" ")
            question = ' '.join(split[1:])

            text_a = question
            inn_split = split[0].split(":")
            label = inn_split[0] + "_" + inn_split[1]
            examples.append((text_a, label))
        f.close()
    return examples

class RoughRuleBase:
    def __init__(self):
        """
        Przechowuje reguły w formie:
        self.rules = [
            {
               "sp_tmp": "Be"/"NB"/"PB"/"CC",
               "rgb":    "Be"/"NB"/"PB"/"CC",
               "depth":  "Be"/"NB"/"PB"/"CC",
               "decision": "O" (object) lub "B" (background)
            },
            ...
        ]
        """
        self.rules = []

    def add_rule(self, sp_tmp, rgb, depth, decision):
        new_rule = {
            "sp_tmp": sp_tmp,
            "rgb": rgb,
            "depth": depth,
            "decision": decision
        }
        self.rules.append(new_rule)

    def classify_granule(self, sp_val, rgb_val, depth_val):
        """
        Przykładowa metoda, która na podstawie atrybutów
        zwraca 'O' lub 'B'.
        """
        for rule in self.rules:
            if (rule["sp_tmp"] == sp_val and
                rule["rgb"]    == rgb_val and
                rule["depth"]  == depth_val):
                return rule["decision"]
        # Jeśli brak dopasowania, np. tło
        return "B"

    def update_rules(self, new_info):
        """
        Gdy flow graph pokazuje duże zmiany, możemy
        zaktualizować pewne reguły na podstawie new_info.
        Tutaj kod jest symboliczny.
        """
        pass

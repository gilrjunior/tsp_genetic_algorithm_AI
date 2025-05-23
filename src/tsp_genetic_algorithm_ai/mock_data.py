from Location import Location

def get_mock_data() -> list[Location]:

    """
    Função que retorna uma lista de locais

    :return: lista de locais
    """
    
    locations = [
        Location(1, "Fazenda em Delta - MG"),
        Location(2, "Zebu Carnes Ramid Mauad"),
        Location(3, "Zebu Carnes Saudade"),
        Location(4, "Zebu Carnes Nossa Senhora do Desterro"),
        Location(5, "Zebu Carnes Governador Magalhães Pinto"),
        Location(6, "Zebu Carnes Erick Silva"),
        Location(7, "Zebu Carnes Conceição das Alagoas"),
        Location(8, "Zebu Carnes Tônico dos Santos"),
        Location(9, "Zebu Carnes José Valim de Melo"),
        Location(10, "Mart Minas Parque Laranjeiras"),
        Location(11, "Mart Minas Olinda"),
        Location(12, "Mart Minas Santos Dumont"),
        Location(13, "Empório Bahamas Praça Uberaba"),
        Location(14, "Bahamas Mix Nossa Senhora do Desterro"),
        Location(15, "Bahamas Mix Santana Borges"),
        Location(16, "Bahamas Mix Cherem"),
        Location(17, "Bahamas Express Leopoldino"),
        Location(18, "Atacadão - Santos Dumont"),
        Location(19, "ABC Atacado e Varejo Cherém"),
        Location(20, "Supermercado ABC - Rua João Alfredo"),
        Location(21, "Villefort")
    ]

    return locations

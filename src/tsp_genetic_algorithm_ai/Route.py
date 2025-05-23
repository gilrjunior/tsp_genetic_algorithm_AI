from Location import Location

class Route:

    def __init__(self, locations):

        """
        Construtor da classe Route

        :param locations: lista de locais
        """

        self.locations = locations

    def __str__(self) -> str:

        """
        Método que retorna a rota

        :return: string com a rota
        """

        route_str = " -> ".join(str(location) for location in self.locations)
        return route_str


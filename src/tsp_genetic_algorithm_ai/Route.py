from .Location import Location
import urllib.parse

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

    def get_google_maps_url(self) -> str:
        """
        Gera uma URL do Google Maps para a rota.

        :return: string com a URL do Google Maps
        """
        # Coordenadas da fazenda em Delta - MG
        farm_coords = "-19.9784413,-47.8070046"
        
        # Lista para armazenar os pontos da rota
        waypoints = []
        
        # Adiciona todos os pontos da rota como waypoints
        for location in self.locations[1:-1]:  # Exclui o primeiro e último ponto (fazenda)
            # Codifica o nome do local para URL
            encoded_name = urllib.parse.quote(location.name)
            waypoints.append(encoded_name)
        
        # Constrói a URL do Google Maps usando as coordenadas da fazenda
        base_url = "https://www.google.com/maps/dir/"
        origin = f"{farm_coords}"
        destination = f"{farm_coords}"
        
        # Adiciona os waypoints à URL
        waypoints_str = "/".join(waypoints)
        
        # Monta a URL final
        url = f"{base_url}{origin}/{waypoints_str}/{destination}"
        
        return url


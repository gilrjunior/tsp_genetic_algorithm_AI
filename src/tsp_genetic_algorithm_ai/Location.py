class Location:

    """
    Classe que representa um local
    """

    def __init__(self, id, name):
        
        """
        Construtor da classe Location

        :param id: int com o id do local
        :param name: string com o nome do local
        """

        self.id = id
        self.name = name

    def __str__(self) -> str:

        """
        Método que retorna o nome do local

        :return: string com o nome do local
        """

        return self.name

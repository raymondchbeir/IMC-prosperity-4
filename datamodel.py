import json
from json import JSONEncoder
from typing import Dict, List, Optional

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        sunlight: float,
        humidity: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight
        self.humidity = humidity


class Observation:
    def __init__(
        self,
        plainValueObservations: Optional[Dict[Product, ObservationValue]] = None,
        conversionObservations: Optional[Dict[Product, ConversionObservation]] = None,
    ):
        self.plainValueObservations = plainValueObservations or {}
        self.conversionObservations = conversionObservations or {}

    def __str__(self) -> str:
        return f"Observation(plain={self.plainValueObservations}, conversion={self.conversionObservations})"


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"

    def __repr__(self) -> str:
        return self.__str__()


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: Optional[UserId] = None,
        seller: Optional[UserId] = None,
        timestamp: int = 0,
    ):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            f"Trade(symbol={self.symbol}, price={self.price}, quantity={self.quantity}, "
            f"buyer={self.buyer}, seller={self.seller}, timestamp={self.timestamp})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TradingState:
    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

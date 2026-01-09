from abc import ABCMeta, abstractmethod

from geoscore_de.address.models import StructAddress


class BaseStructAddressRetriever(metaclass=ABCMeta):
    @abstractmethod
    def get_struct_address(self, raw_address: str) -> StructAddress | None:
        """Get structured address from raw address string.

        Args:
            raw_address (str): Raw address string.

        Returns:
            StructAddress | None: Structured address or None if not found.
        """
        pass

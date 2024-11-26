from src.database.IPT1_reader import IPT1Reader
from src.database.database_config import DatabaseConfig


class Main:
    def main():
        """Runs the experiemnt"""
        config = DatabaseConfig()
        reader = IPT1Reader(config)

        try: 
            df = reader.read_ipt1_data()
            print("Data loaded sucessfully")
            print(f"DataFrame shape: {df.shape}")
        except Exception as e:
            print(f"Failed to read data: {str(e)}")
    if __name__ == "__main__":
        main()
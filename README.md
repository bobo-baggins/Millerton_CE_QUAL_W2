
# CeQUAL Model 
2024-07-07

This is a suite of python scripts to assit in running the core CE-QUAL-W2 model; including pre and post processing tools. 

The core model is run using the CE-QUAL-W2 executable through the script 'CeQUAL_model\run_simulations.py'.

Reccomended method is setting up a virtual environment and using the provided 'req.txt' file to install all required packages.

Instructions below

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.x

### Setting Up the Virtual Environment

1. Open Windows PowerShell.
2. Navigate to your project directory:

    ```powershell
    cd path\to\your\project
    ```

3. Create a virtual environment:

    ```powershell
    python -m venv venv
    ```

4. Activate the virtual environment:

    ```powershell
    .\venv\Scripts\Activate
    ```

### Installing Required Packages

Required Python packages are listed in `req.txt`. Install them using pip:

```powershell
pip install -r req.txt
```

## Usage

1. Set the time in `\CeQUAL_model\w2_con.npt`. Line 28, file is space delinated and date is relative to a Julian Day of 1 on January 1st, 1921. This is the extent of records included in the model.

2. Update the time in the core script `\CeQUAL_model\run_simulations.py` starting on Line 24. 

3. Run the core script:

    ```powershell
    python .\CeQUAL_model\run_simulations.py
    ```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Justin McBain - [Justin@McBainAssocaties.com](mailto:Justin@McBainAssocaties.com)

Project Link: [https://github.com/yourusername/yourproject](https://github.com/yourusername/yourproject)
```

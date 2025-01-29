# Bank Accounts Management System

## Description

The **Bank Accounts Management System** is designed to efficiently manage customer accounts and handle various banking operations such as account types, loans, and transactions. This system handles different types of bank accounts, including saving, current, investment deposits, corporate accounts, and loans. It also tracks customer information and the relationships between customers and their accounts.

### Project Requirements

- **Customer Information**:
  - First Name (FName), Middle Name (MName), Last Name (LName)
  - Nationality, Personal Document Number (PDNum), Residence Proof Document Number (ResProofDoc), Phone Number (PhoneNum)

- **Bank Account Information**:
  - Account Number (AccountNum): Automatically generated, formatted internationally
  - Balance (Balance)
  - Account Types:
    - **Saving Account**:
      - Multiple currencies (JOD, EUR, USD, GBP)
      - ATM Deposit Card Number (CardNum)
      - Profit percentage based on the smallest deposit, calculated at 0.375% every 3 months
    - **Current Account**:
      - Multiple currencies (JOD, EUR, USD, GBP)
      - Deposit Card Number (DCardNum), Credit Card Number (CCardNum)
      - Bank Check Book Number (CheckBookNum), Number of checks
    - **Investments Deposit Account**:
      - One currency (JOD or USD)
      - Deposit Period (DPeriod) in days, profit calculated by the formula:
        `((Balance × ProfitRate × NoOfDays) / 365)`
    - **Corporate Account**:
      - Corporate Name (CorName), Corporate Registration Number (CorRegNumber)
    - **Loan**:
      - Associated with an Account Number (AccountNum)
      - Calculates Maximum Loan (MaxLoan), Loan Amount (LoanAmount), Monthly Repayment (MonAmount), and Delays (Delays)

- **Relationships**:
  - A customer can have multiple accounts.
  - An account can have multiple loans associated with it.
  - A corporate account can be linked to multiple customers (shareholders).
  - Customers can withdraw and deposit funds from any of their accounts.

## Steps for Designing the Database

### Step 1: Requirements Collection
- Gathered requirements for the application, covering all relevant customer, account, and loan details.

### Step 2: ER/EER Diagram
- Created an Entity-Relationship (ER) or Enhanced Entity-Relationship (EER) diagram that illustrates the entities, their attributes, relationships, and cardinalities.

### Step 3: Relational Schema
- Mapped the ER/EER diagram into a relational schema, creating tables with foreign key constraints and relationships.

### Step 4: Database Implementation
- Created database tables using Oracle, specifying primary keys, foreign keys, and constraints. The tables were populated with sample data, and SQL statements were written for database manipulation and queries.

### Step 5: Front-end Interface (Bonus)
- Created a front-end interface with forms and reports to interact with the database.

## Documentation

### 1. Introduction
- **Application Description**: This system manages bank accounts, transactions, and user information in a banking environment.
- **Reason for Choosing**: A bank management system is a practical and useful application of database design, helping manage sensitive financial data securely.

### 2. Application Requirements
- **User Information**: Personal data of account holders, such as name, address, and contact details.
- **Account Information**: Types of bank accounts, balances, and account status.
- **Transaction Information**: Details of deposits and withdrawals.
- **Branch Information**: Locations and details of bank branches.

### 3. ER/EER Diagram
- A visual representation of the system’s entities and their relationships.

### 4. Relational Schema
- The schema includes all tables, their attributes, and the relationships between them.

### 5. Sample Database Instance
- A sample of records (at least five rows) for each table in the database.

### 6. SQL Statements
- **Creating Relations**: SQL statements to create tables with constraints.
- **Inserting Records**: SQL statements to insert initial data.
- **Modifying Records**: SQL statements to update records.
- **Deleting Records**: SQL statements to delete records.
- **Retrieving Data**: SQL queries to retrieve records using joins and aggregate functions.

## Technologies Used
- **Database**: Oracle
- **Tools**: ERDPlus for diagram creation, SQL for database manipulation

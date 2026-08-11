"""Create a sample SQLite database with multiple tables for testing."""
import sqlite3
import random
import numpy as np

random.seed(42)
np.random.seed(42)

conn = sqlite3.connect("sample_company.db")
cursor = conn.cursor()

# Create departments table
cursor.execute("""
CREATE TABLE IF NOT EXISTS departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    budget REAL,
    headcount_target INTEGER
)
""")

# Create employees table
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    emp_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    dept_id INTEGER,
    age INTEGER,
    years_experience REAL,
    education_level TEXT,
    salary REAL,
    performance_score REAL,
    satisfaction_score REAL,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
)
""")

# Create sales table
cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY,
    emp_id INTEGER,
    product_category TEXT,
    units_sold INTEGER,
    unit_price REAL,
    sale_date TEXT,
    region TEXT,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
)
""")

# Insert departments
departments = [
    (1, "Engineering", 500000, 50),
    (2, "Sales", 300000, 40),
    (3, "Marketing", 200000, 25),
    (4, "HR", 100000, 15),
    (5, "Finance", 150000, 20),
]
cursor.executemany("INSERT OR REPLACE INTO departments VALUES (?,?,?,?)", departments)

# Generate employees
education_levels = ["High School", "Bachelor", "Master", "PhD"]
names = [f"Employee_{i}" for i in range(1, 201)]
employees = []
for i in range(1, 201):
    dept_id = random.randint(1, 5)
    age = random.randint(22, 60)
    years_exp = round(random.uniform(0, 35), 1)
    edu = random.choice(education_levels)
    # Salary depends on experience, education, and department
    base_salary = 40000 + years_exp * 2000
    if edu == "Bachelor":
        base_salary += 10000
    elif edu == "Master":
        base_salary += 20000
    elif edu == "PhD":
        base_salary += 30000
    if dept_id == 1:
        base_salary *= 1.3
    elif dept_id == 2:
        base_salary *= 1.1
    salary = round(base_salary + np.random.normal(0, 5000), 2)
    performance = round(np.clip(np.random.normal(70, 15), 30, 100), 1)
    satisfaction = round(np.clip(np.random.normal(65, 20), 20, 100), 1)
    employees.append((i, names[i-1], dept_id, age, years_exp, edu, salary, performance, satisfaction))

cursor.executemany("INSERT OR REPLACE INTO employees VALUES (?,?,?,?,?,?,?,?,?)", employees)

# Generate sales data
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Food", "Furniture", "Books"]
sales = []
sale_id = 1
for emp_id in range(1, 201):
    num_sales = random.randint(5, 20)
    for _ in range(num_sales):
        category = random.choice(categories)
        units = random.randint(1, 50)
        price = round(random.uniform(10, 500), 2)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = f"2024-{month:02d}-{day:02d}"
        region = random.choice(regions)
        sales.append((sale_id, emp_id, category, units, price, date, region))
        sale_id += 1

cursor.executemany("INSERT OR REPLACE INTO sales VALUES (?,?,?,?,?,?,?)", sales)

conn.commit()
conn.close()

print("Sample database 'sample_company.db' created successfully!")
print(f"  - departments: {len(departments)} rows")
print(f"  - employees: {len(employees)} rows")
print(f"  - sales: {len(sales)} rows")
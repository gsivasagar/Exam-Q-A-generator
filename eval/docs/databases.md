# Databases and SQL

## Relational Model
A relational database stores data in tables made of rows and columns. A primary key uniquely identifies each row in a table and cannot be null. A foreign key is a column that references the primary key of another table and enforces referential integrity. Normalization removes redundancy, and third normal form requires that every non-key column depends only on the key. Denormalization deliberately duplicates data to speed up reads.

## Querying
SELECT retrieves rows and WHERE filters them before any grouping happens. An INNER JOIN returns only the rows with matching keys in both tables, while a LEFT JOIN keeps every row from the left table and fills missing matches with NULL. GROUP BY collapses rows into groups so that aggregate functions such as COUNT, SUM and AVG can be applied. The HAVING clause filters groups after aggregation, unlike WHERE which filters individual rows. A subquery is a query nested inside another query, and a common table expression names a subquery with the WITH keyword.

## Transactions
A transaction is a group of operations that either all succeed or all fail. ACID stands for atomicity, consistency, isolation and durability. Isolation levels such as read committed and serializable control which anomalies concurrent transactions can see, including dirty reads and phantom reads. A deadlock occurs when two transactions each wait for a lock the other holds, and the database resolves it by aborting one of them. Write-ahead logging records changes in a log before applying them so the database can recover after a crash.

## Indexing and Scaling
An index speeds up lookups by keeping a sorted structure, usually a B-tree, over one or more columns. Indexes make reads faster but slow down inserts and updates because the index must be maintained. A query plan produced by the optimizer shows whether a table scan or an index seek will be used. NoSQL document stores such as MongoDB hold flexible JSON-like records and scale horizontally by sharding. The CAP theorem states that a distributed system cannot guarantee consistency, availability and partition tolerance at once.

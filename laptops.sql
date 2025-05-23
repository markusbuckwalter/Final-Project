--
-- PostgreSQL database dump
--

-- Dumped from database version 13.3
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: laptops; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.laptops (
    brand text,
    laptop_name text,
    display_size numeric,
    processor_type text,
    graphics_card text,
    disk_space text,
    discount_price numeric,
    old_price numeric,
    ratings_5max text
);


--
-- Data for Name: laptops; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.laptops (brand, laptop_name, display_size, processor_type, graphics_card, disk_space, discount_price, old_price, ratings_5max) FROM stdin;
HP	Notebook 14-df0008nx	14.0	 Intel Celeron N4000	 Intel HD Graphics 600	 64 GB (eMMC)	1259.0	1259.0	0 / 5
Lenovo	IdeaPad 330S-14IKB	14.0	 Intel Core i5-8250U	 Intel UHD Graphics 620	 1 TB HDD	1849.0	2099.0	3.3 / 5
Huawei	MateBook D Volta	14.0	 Intel Core i5-8250U	 NVIDIA GeForce MX150 (2 GB)	 256 GB SSD	2999.0	3799.0	0 / 5
Dell	Inspiron 15 3567	15.6	 Intel Core i3-7020U	 Intel HD Graphics 620	 1 TB HDD	1849.0	1849.0	0 / 5
Asus	VivoBook 15 X510UR	15.6	 Intel Core i7-8550U	 NVIDIA GeForce 930MX (2 GB)	 1 TB HDD	2499.0	3149.0	0 / 5
Dell	Vostro 5471	14.0	 Intel Core i7-8550U	 AMD Radeon 530 (4 GB)	 128 GB SSD/1 TB HDD	3799.0	3799.0	3.8 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 128 GB (PCIe SSD)	4649.0	5199.0	4.0 / 5
Huawei	MateBook D	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX150 (2 GB)	 128 GB SSD/1 TB HDD	2449.0	2799.0	4.4 / 5
Huawei	MateBook X Pro	13.88	 Intel Core i5-8250U	 NVIDIA GeForce MX150 (2 GB)	 256 GB NVMe M.2 SSD	4999.0	5999.0	0 / 5
HP	14-cf0007nx	14.0	 Intel Core i5-8250U	 AMD Radeon 530 (2 GB)	 16 GB (Optane)/1 TB HDD	2629.0	2629.0	0 / 5
HP	15-db0001nx	15.6	 AMD A9-9425	 AMD Radeon R5	 1 TB HDD	1999.0	1999.0	0 / 5
Acer	Swift 5	14.0	 Intel Core i7-8565U	 Intel GMA HD	 512 GB SSD	4499.0	5999.0	0 / 5
Apple	MacBook Air	13.3	 Intel Core i5 Dual Core	 Intel HD Graphics 6000	 128 GB (PCIe Flash)	3399.0	4499.0	0 / 5
Acer	Swift 5 SF514-52TP-8933	14.0	 Intel Core i7-8550U	 Intel UHD Graphics 620	 512 GB SSD	4899.0	5999.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i7 6 Core	 Radeon Pro 555X GDDR5 (4 GB)	 256 GB SSD	9099.0	10199.0	0 / 5
Acer	Swift 3 SF314-52G	14.0	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 512 GB SSD	3399.0	4399.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 655	 512 GB SSD	7599.0	8499.0	4.4 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 655	 256 GB SSD	6969.0	7769.0	4.3 / 5
Huawei	MateBook 13	13.0	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 512 GB SSD	4999.0	4999.0	4.1 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 256 GB (PCIe SSD)	5399.0	6099.0	3.9 / 5
Dell	XPS 13 9380	13.3	 Intel Core i5-8265U	 Intel UHD Graphics 620	 256 GB PCIe NVMe M.2 SSD	4699.0	5399.0	4.4 / 5
HP	Pavilion 13-an0001nx	13.3	 Intel Core i7-8565U	 Intel UHD Graphics 620	 256 GB PCIe NVMe M.2 SSD	3299.0	3999.0	0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 256 GB (PCIe SSD)	5449.0	6099.0	4.0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 256 GB (PCIe SSD)	5449.0	6099.0	3.9 / 5
HP	Pavilion 14-ce0001nx	14.0	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (4 GB)	 256 GB M.2 SSD/1 TB HDD	3599.0	4199.0	0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 128 GB (PCIe SSD)	4599.0	5199.0	3.9 / 5
HP	Pavilion 14-ce0000nx	14.0	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (4 GB)	 256 GB M.2 SSD/1 TB HDD	3599.0	4199.0	5.0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 128 GB (PCIe SSD)	4599.0	5199.0	3.9 / 5
Huawei	MateBook D Volta	14.0	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 256 GB SSD	3599.0	4199.0	0 / 5
Asus	ZenBook 14 UX433FN	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 512 GB SSD	4299.0	4899.0	0 / 5
Asus	VivoBook 14 S430FN	14.0	 Intel Core i5-8265U	 NVIDIA GeForce MX150 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2599.0	3199.0	0 / 5
Huawei	MateBook X Pro	13.88	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 512 GB SSD	6399.0	6999.0	0 / 5
HP	Pavilion 14-ce2003nx	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX250 (4 GB)	 128 GB M.2 SSD/1 TB HDD	3249.0	3799.0	0 / 5
Asus	X543UB	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX110 (2 GB)	 1 TB HDD	2499.0	2999.0	0 / 5
HP	Pavilion 14-ce2000nx	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX250 (4 GB)	 256 GB NVMe M.2 SSD/1 TB HDD	3799.0	4299.0	0 / 5
HP	Pavilion 14-ce2001nx	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX250 (4 GB)	 256 GB NVMe M.2 SSD/1 TB HDD	3799.0	4299.0	0 / 5
Lenovo	IdeaPad 330S-14IKB	14.0	 Intel Core i5-8250U	 Intel UHD Graphics 620	 256 GB (PCIe Flash)	1999.0	2499.0	0 / 5
Asus	VivoBook S15 S530FN	15.6	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 128 GB M.2 SSD/1 TB HDD	3499.0	3999.0	0 / 5
Asus	VivoBook 15 X512UB	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX110 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2399.0	2899.0	0 / 5
Asus	VivoBook 14 X412UB	14.0	 Intel Core i5-8250U	 NVIDIA GeForce MX110 (2 GB)	 1 TB HDD	2299.0	2799.0	0 / 5
Asus	X507UB	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX110 (2 GB)	 1 TB HDD	1999.0	2449.0	3.6 / 5
Asus	VivoBook 14 X412FJ	14.0	 Intel Core i5-8265U	 NVIDIA GeForce MX230 (2 GB)	 128 GB SSD/1 TB HDD	2649.0	3099.0	0 / 5
Apple	MacBook	12.0	 Intel Core M3	 Intel HD Graphics 615	 256 GB SSD	5269.0	5669.0	5.0 / 5
Huawei	MateBook D	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 128 GB SSD/1 TB HDD	2899.0	3299.0	4.0 / 5
Huawei	MateBook D	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 128 GB SSD/1 TB HDD	2899.0	3299.0	4.4 / 5
Huawei	MateBook 13	13.0	 Intel Core i7-8565U	 Intel UHD Graphics 620	 512 GB SSD	3599.0	3999.0	4.1 / 5
Lenovo	IdeaPad S530	13.3	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 512 GB NVMe M.2 SSD	3599.0	3999.0	0 / 5
Dell	XPS 13 9380	13.3	 Intel Core i7-8565U	 Intel UHD Graphics 620	 512 GB PCIe NVMe M.2 SSD	5999.0	6399.0	4.4 / 5
HP	15-da1019nx	15.6	 Intel Core i7-8565U	 NVIDIA GeForce MX130 (4 GB)	 1 TB HDD	2749.0	3149.0	0 / 5
HP	15-da1003nx	15.6	 Intel Core i7-8565U	 NVIDIA GeForce MX130 (4 GB)	 1 TB HDD	2749.0	3149.0	0 / 5
Dell	XPS 15 9570	15.6	 Intel Core i7-8750H	 NVIDIA GeForce GTX 1050 Ti (4 GB)	 128 GB M.2 SSD/1 TB HDD	6099.0	6499.0	0 / 5
Asus	VivoBook 15 X512UF	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX130 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2899.0	3299.0	0 / 5
Dell	Inspiron 15 3580	15.6	 Intel Core i5-8265U	 AMD Radeon 520 (2 GB)	 1 TB HDD	2299.0	2699.0	0 / 5
Acer	Aspire 5 A515-52G-75XJ	15.6	 Intel Core i7-8565U	 NVIDIA GeForce MX130 (2 GB)	 1 TB HDD	2799.0	3199.0	0 / 5
Dell	Inspiron 15 3580	15.6	 Intel Core i7-8565U	 AMD Radeon 520 (2 GB)	 1 TB HDD	2799.0	3149.0	0 / 5
Huawei	MateBook D	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX150 (2 GB)	 128 GB SSD/1 TB HDD	2449.0	2799.0	0 / 5
Lenovo	IdeaPad 330	15.6	 Intel Core i7-8550U	 AMD Radeon 530 (4 GB)	 128 GB M.2 SSD/1 TB HDD	2789.0	3099.0	3.6 / 5
Lenovo	IdeaPad S340	15.6	 Intel Core i5-8265U	 NVIDIA GeForce MX230 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2299.0	2599.0	0 / 5
HP	Pavilion 15-cs2003nx	15.6	 Intel Core i5-8265U	 NVIDIA GeForce MX130 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2699.0	2999.0	0 / 5
Acer	Swift 3 314	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 128 GB SSD/1 TB HDD	3699.0	3999.0	0 / 5
Lenovo	IdeaPad S340	14.0	 Intel Core i3-8145U	 Intel GMA HD	 1 TB HDD	1499.0	1779.0	4.2 / 5
Lenovo	IdeaPad S340	14.0	 Intel Core i3-8145U	 Intel GMA HD	 1 TB HDD	1499.0	1779.0	4.2 / 5
Lenovo	IdeaPad S145	15.6	 Intel Celeron 4205U	 Intel GMA HD	 500 GB HDD	1039.0	1299.0	0 / 5
Dell	Inspiron 15 3580	15.6	 Intel Core i5-8265U	 AMD Radeon 520 (2 GB)	 1 TB HDD	2199.0	2449.0	0 / 5
Lenovo	IdeaPad S340	14.0	 Intel Core i5-8265U	 Intel GMA HD	 1 TB HDD	1849.0	2099.0	4.2 / 5
HP	14-cf1001nx	14.0	 Intel Core i5-8265U	 AMD Radeon 530 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2549.0	2799.0	5.0 / 5
HP	15-da1005nx	15.6	 Intel Core i5-8265U	 Intel UHD Graphics 620	 1 TB HDD	1949.0	2199.0	0 / 5
Lenovo	IdeaPad S340	14.0	 Intel Core i5-8265U	 Intel GMA HD	 1 TB HDD	1849.0	2099.0	4.2 / 5
Lenovo	IdeaPad 330-15IKBR	15.6	 Intel Core i7-8550U	 AMD Radeon 530M (4 GB)	 2 TB HDD	2899.0	3149.0	0 / 5
Dell	Inspiron 15 3581	15.6	 Intel Core i3-7020U	 Intel HD Graphics 620	 1 TB HDD	1699.0	1929.0	0 / 5
Acer	Aspire 3 A315-53-52ZL	15.6	 Intel Core i5-8250U	 Intel UHD Graphics 620	 16 GB (Optane)/1 TB HDD	1979.0	2199.0	0 / 5
Acer	Aspire 3 A315-53-341N	15.6	 Intel Core i3-7020U	 Intel Graphics 620	 1 TB HDD	1539.0	1749.0	0 / 5
Acer	Aspire 3 A315-53G	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX130 (2 GB)	 1 TB HDD	2299.0	2499.0	0 / 5
Lenovo	IdeaPad 330	15.6	 Intel Celeron N4000	 Intel UHD Graphics 600	 1 TB HDD	1179.0	1379.0	0 / 5
Lenovo	IdeaPad S130-14IGM	14.0	 Intel Celeron N4000	 Intel UHD Graphics 600	 64 GB (eMMC)	899.0	1099.0	0 / 5
Lenovo	IdeaPad 330S	15.6	 Intel Core i5-8250U	 AMD Radeon 540 (4 GB)	 2 TB HDD	2499.0	2699.0	0 / 5
Asus	VivoBook 14 X420UA	14.0	 Intel Core i3-7020U	 Intel HD Graphics 620	 128 GB M.2 SSD	1499.0	1699.0	0 / 5
Asus	X540UB	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX110 (2 GB)	 1 TB HDD	1799.0	1999.0	0 / 5
Huawei	MateBook 13	13.0	 Intel Core i5-8265U	 Intel UHD Graphics 620	 256 GB SSD	2799.0	2999.0	4.1 / 5
HP	15-da1018nx	15.6	 Intel Core i3-8145U	 Intel UHD Graphics 620	 1 TB HDD	1629.0	1799.0	0 / 5
HP	15-da1016nx	15.6	 Intel Core i3-8145U	 Intel UHD Graphics 620	 1 TB HDD	1629.0	1799.0	0 / 5
Acer	Aspire 3 A315-33-C6S9	15.6	 Intel Celeron N3060	 Intel GMA HD	 500 GB HDD	1219.0	1369.0	3.7 / 5
Asus	X543UB	15.6	 Intel Core i5-8250U	 Intel UHD Graphics 620	 1 TB HDD	1799.0	1949.0	0 / 5
Asus	X543MA	15.6	 Intel Celeron N4000	 Intel UHD Graphics 600	 1 TB HDD	1099.0	1249.0	0 / 5
Acer	Aspire 3 A315-53	15.6	 Intel Core i3-7020U	 Intel HD Graphics 620	 1 TB HDD	1599.0	1749.0	3.7 / 5
Dell	Inspiron 15 3582	15.6	 Intel Celeron N4000	 Intel UHD Graphics 600	 500 GB HDD	1299.0	1399.0	0 / 5
Dell	Inspiron 15 3573	15.6	 Intel Celeron N4000	 Intel GMA HD	 500 GB HDD	1299.0	1399.0	0 / 5
Dell	Inspiron 15 5584	15.6	 Intel Core i7-8565U	 NVIDIA GeForce MX130 (4 GB)	 256 GB SSD/1 TB HDD	3999.0	4099.0	3.7 / 5
Acer	Swift 1 SF114-32-C4GB	14.0	 Intel Celeron N4000	 Intel HD Graphics 600	 64 GB (eMMC)	1279.0	1349.0	0 / 5
Acer	Aspire 3	15.6	 Intel Core i3-7020U	 Intel HD Graphics 620	 1 TB HDD	1679.0	1749.0	0 / 5
Acer	Aspire 3 A315-54	15.6	 Intel Core i5-8265U	 Intel UHD Graphics 620	 1 TB HDD	1999.0	1999.0	0 / 5
Acer	Aspire 3 A315-54	15.6	 Intel Core i5-8265U	 Intel UHD Graphics 620	 1 TB HDD	1999.0	1999.0	0 / 5
Acer	Aspire 5 A515-54G	15.6	 Intel Core i7-8565U	 NVIDIA GeForce MX250 (2 GB)	 1 TB HDD	3099.0	3099.0	0 / 5
HP	Pavilion 15-cs2012nx	15.6	 Intel Core i7-8565U	 NVIDIA GeForce GTX 1050 (3 GB)	 512 GB NVMe M.2 SSD	4549.0	4549.0	0 / 5
Asus	ZenBook S13 UX392FN	13.9	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 1 TB PCIe NVMe M.2 SSD	6999.0	6999.0	0 / 5
HP	Pavilion 15-cs2011nx	15.6	 Intel Core i5-8265U	 NVIDIA GeForce MX130 (2 GB)	 1 TB HDD	2799.0	2799.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 645	 128 GB SSD	5649.0	5649.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 645	 128 GB SSD	5649.0	5649.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 645	 256 GB SSD	6499.0	6499.0	0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 128 GB (PCIe SSD)	4799.0	4799.0	3.9 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 128 GB (PCIe SSD)	4799.0	4799.0	3.9 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 128 GB (PCIe SSD)	4799.0	4799.0	0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 256 GB (PCIe SSD)	5649.0	5649.0	0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 256 GB (PCIe SSD)	5649.0	5649.0	0 / 5
Apple	MacBook Air (Retina)	13.3	 Intel Core i5 Dual Core	 Intel UHD Graphics 617	 256 GB (PCIe SSD)	5649.0	5649.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 645	 256 GB SSD	6499.0	6499.0	0 / 5
HP	Stream 14-cb003nx	14.0	 Intel Celeron N3060	 Intel HD Graphics 400	 32 GB (eMMC)	1049.0	1049.0	0 / 5
Apple	MacBook Pro (Retina)	13.3	 Intel Core i5 Dual Core	 Intel Iris Plus Graphics 640	 128 GB SSD	5669.0	5669.0	4.9 / 5
Acer	Aspire 1 A114-31-C6WP	14.0	 Intel Celeron N3350	 Intel HD Graphics 505	 64 GB (eMMC)	999.0	999.0	0 / 5
Microsoft	Surface	13.5	 Intel Core i5-7200U	 Intel HD Graphics 620	 256 GB SSD	5799.0	5799.0	0 / 5
HP	ENVY 13-ah0006nx	13.3	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 512 GB PCIe NVMe M.2 SSD	4939.0	4939.0	0 / 5
Acer	Swift 1 SF114-32-C4GB	14.0	 Intel Celeron N4000	 Intel HD Graphics 600	 64 GB (eMMC)	1349.0	1349.0	0 / 5
Acer	Aspire 1 A114-31	14.0	 Intel Celeron N3350	 Intel HD Graphics 500	 64 GB (eMMC)	999.0	999.0	0 / 5
Dell	XPS 13 9370	13.3	 Intel Core i7-8550U	 Intel UHD Graphics 620	 1 TB SSD	8099.0	8099.0	0 / 5
Dell	Vostro 5471	14.0	 Intel Core i5-8250U	 AMD Radeon 530 (2 GB)	 1 TB HDD	2949.0	2949.0	0 / 5
HP	Spectre 13-af001nx	13.3	 Intel Core i5-8250U	 Intel UHD Graphics 620	 512 GB (PCIe Flash)	6299.0	6299.0	0 / 5
Dell	XPS 13 9380	13.3	 Intel Core i7-8565U	 Intel UHD Graphics 620	 1 TB SSD	8099.0	8099.0	4.4 / 5
HP	ENVY 13-ah0001nx	13.3	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 512 GB PCIe NVMe M.2 SSD	4939.0	4939.0	4.0 / 5
Apple	MacBook Pro (Retina)	13.3	 Intel Core i5 Dual Core	 Intel Iris Plus Graphics 640	 256 GB SSD	6499.0	6499.0	4.6 / 5
Lenovo	IdeaPad 330-15IGM	15.6	 Intel Celeron N4000	 Intel HD Graphics 600	 500 GB HDD	1359.0	1359.0	3.9 / 5
HP	ENVY 13-ah0002nx	13.3	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 1 TB PCIe NVMe M.2 SSD	6299.0	6299.0	0 / 5
Dell	XPS 13 9360	13.3	 Intel Core i5-8250U	 Intel GMA HD	 256 GB PCIe NVMe M.2 SSD	4899.0	4899.0	4.3 / 5
Dell	Vostro 5471	14.0	 Intel Core i5-8250U	 AMD Radeon 530 (4 GB)	 128 GB SSD/1 TB HDD	3599.0	3599.0	3.8 / 5
Dell	Inspiron 13 5370	13.3	 Intel Core i5-8250U	 AMD Radeon 530 (2 GB)	 256 GB SSD	3149.0	3149.0	0 / 5
Acer	Aspire 3 A315-53	15.6	 Intel Core i5-7200U	 NVIDIA GeForce MX130 (2 GB)	 1 TB HDD	2249.0	2249.0	3.7 / 5
Dell	Vostro 5481	14.0	 Intel Core i5-8265U	 NVIDIA GeForce MX130 (2 GB)	 128 GB PCIe NVMe M.2 SSD/1 TB HDD	3399.0	3399.0	0 / 5
Dell	Vostro 5481	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX130 (2 GB)	 128 GB PCIe NVMe M.2 SSD/1 TB HDD	3899.0	3899.0	0 / 5
Microsoft	Surface 2	13.5	 Intel Core i5-8250U	 Intel UHD Graphics 620	 128 GB SSD	4199.0	4199.0	4.1 / 5
Microsoft	Surface 2	13.5	 Intel Core i5-8250U	 Intel UHD Graphics 620	 256 GB SSD	5399.0	5399.0	0 / 5
HP	Stream 14-cb104nx	14.0	 Intel Celeron N4000	 Intel UHD Graphics 600	 64 GB (eMMC)	1099.0	1099.0	0 / 5
Dell	Inspiron 14 5480	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX250 (2 GB)	 128 GB SSD/1 TB HDD	4099.0	4099.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i7 6 Core	 Radeon Pro 555X GDDR5 (4 GB)	 256 GB SSD	10799.0	10799.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i9	 Radeon Pro 560X GDDR5 (4 GB)	 512 GB SSD	12499.0	12499.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 655	 256 GB SSD	7999.0	7999.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Quad Core	 Intel Iris Plus Graphics 655	 512 GB SSD	8699.0	8699.0	0 / 5
MSI	GT83VR 7RF Titan SLI	18.4	 Intel Core i7-7820HK	 NVIDIA GeForce GTX 1080 (8 GB) SLI	 256 GB SSD (Super Raid)/1 TB HDD	9071.0	9071.0	0 / 5
Asus	ROG G752VS	17.3	 Intel Core i7-7700HQ	 NVIDIA GeForce GTX 1070 (8 GB)	 256 GB SSD/1 TB HDD	8189.0	8189.0	0 / 5
Acer	Predator 17 GX791 78ND	17.3	 Intel Core i7-6700HQ	 NVIDIA GeForce GTX 980M (8 GB)	 256 GB SSD/1 TB HDD	7507.0	7507.0	0 / 5
Apple	MacBook Pro (Retina)	15.4	 Intel Core i7 Quad Core	 AMD Radeon R9-M370X (2 GB)	 512 GB SSD	6824.0	6824.0	0 / 5
Dell	Alienware 15	15.6	 Intel Core i7-7820HK	 NVIDIA GeForce GTX 1080 (8 GB)	 256 GB SSD/1 TB HDD	6688.0	6688.0	3.7 / 5
Dell	XPS 15	15.6	 Intel Core i7-7700HQ	 NVIDIA GeForce GTX 1050 (4 GB)	 1 TB HDD	6483.0	6483.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i7 Quad Core	 Radeon Pro 555 GDDR5 (2 GB)	 256 GB SSD	7679.0	7679.0	4.6 / 5
HP	ENVY 13-ab002nx	13.3	 Intel Core i7-7500U	 Intel HD Graphics 620	 1 TB PCIe NVMe M.2 SSD	4249.0	4249.0	0 / 5
Asus	ZenBook UX433FN	14.0	 Intel Core i7-8565U	 NVIDIA GeForce MX150 (2 GB)	 512 GB SSD	3799.0	4999.0	4.4 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i7 6 Core	 Radeon Pro 560X GDDR5 (4 GB)	 512 GB SSD	10699.0	11899.0	4.4 / 5
HP	ENVY 13-ad002nx	13.3	 Intel Core i7-7500U	 Intel Graphics 620	 512 GB PCIe NVMe M.2 SSD	3539.0	3539.0	4.1 / 5
Apple	MacBook	12.0	 Intel Core M3	 Intel HD Graphics 615	 256 GB SSD	4669.0	5669.0	5.0 / 5
HP	Pavilion 15-cs0006nx	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (4 GB)	 128 GB M.2 SSD/1 TB HDD	2979.0	3779.0	3.9 / 5
Asus	VivoBook 15 X510UF	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX130 (2 GB)	 128 GB M.2 SSD/1 TB HDD	2599.0	3299.0	0 / 5
Asus	VivoBook S430	14.0	 Intel Core i5-8250U	 NVIDIA GeForce MX130 (2 GB)	 256 GB M.2 SSD/1 TB HDD	2499.0	3199.0	0 / 5
HP	Pavilion 15-cs0001nx	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX130 (2 GB)	 16 GB (Optane)/1 TB HDD	2409.0	3049.0	0 / 5
HP	Pavilion 15-cs0000nx	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX130 (2 GB)	 16 GB (Optane)/1 TB HDD	2409.0	3049.0	0 / 5
Acer	Swift 3 SF314-54G-87HB	14.0	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 1 TB HDD	3399.0	3899.0	0 / 5
Asus	VivoBook 15 X507UB	15.6	 Intel Core i5-8250U	 NVIDIA GeForce MX110 (2 GB)	 128 GB SSD/1 TB HDD	2099.0	2599.0	0 / 5
HP	15-bs150nx	15.6	 Intel Core i3-5005U	 Intel GMA HD	 500 GB HDD	1199.0	1199.0	0 / 5
Dell	Inspiron 3567	15.6	 Intel Core i3-6006U	 Intel GMA HD	 1 TB HDD	1429.0	1429.0	0 / 5
Apple	MacBook Pro (Retina)	13.3	 Intel Core i5 Dual Core	 Intel Iris Plus Graphics 640	 128 GB SSD	5269.0	5669.0	4.6 / 5
Lenovo	IdeaPad 330S-15IKB	15.6	 Intel Core i5-8250U	 AMD Radeon 535 (2 GB)	 1 TB HDD	2169.0	2519.0	4.0 / 5
Acer	Aspire 3 A315-53-34CE	15.6	 Intel Core i3-7020U	 Intel Graphics 620	 1 TB HDD	1239.0	1549.0	4.0 / 5
Asus	VivoBook 15 X540UA	15.6	 Intel Core i5-8250U	 Intel HD Graphics 620	 1 TB HDD	1599.0	1899.0	5.0 / 5
Lenovo	IdeaPad 330-15IGM	14.0	 Intel Core i3-8130U	 Intel HD Graphics 620	 2 TB HDD	1589.0	1889.0	3.6 / 5
Lenovo	IdeaPad 330S-14IKB	14.0	 Intel Core i5-8250U	 Intel UHD Graphics 620	 1 TB HDD	1849.0	2099.0	3.3 / 5
Lenovo	IdeaPad 330S-14IKB	14.0	 Intel Core i3-8130U	 Intel HD Graphics 620	 2 TB HDD	1699.0	1889.0	3.3 / 5
HP	14-bp101nx	14.0	 Intel Core i5-8250U	 AMD Radeon 530 (2 GB)	 1 TB HDD	2519.0	2519.0	5.0 / 5
Apple	MacBook Pro (Retina)	15.4	 Intel Core i7 Quad Core	 Intel Iris Pro Graphics	 256 GB SSD	8189.0	8189.0	4.5 / 5
HP	14-ck0008nx	14.0	 Intel Celeron N4000	 Intel HD Graphics 600	 500 GB HDD	1369.0	1369.0	0 / 5
HP	15-da0007nx	15.6	 Intel Core i3-7020U	 Intel HD Graphics 620	 1 TB HDD	1779.0	1779.0	5.0 / 5
HP	15-da0035nx	15.6	 Intel Core i7-8550U	 Intel UHD Graphics 620	 1 TB HDD	2299.0	2299.0	3.0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Dual Core	 Intel Iris Plus Graphics 650	 512 GB SSD	8499.0	8499.0	0 / 5
HP	Pavilion 15-cs0003nx	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (4 GB)	 128 GB M.2 SSD/1 TB HDD	3779.0	3779.0	0 / 5
HP	15-bs007nx	15.6	 Intel Core i3-6006U	 Intel HD Graphics 520	 1 TB HDD	1779.0	1779.0	0 / 5
Lenovo	IdeaPad 320-15IKBRA	15.6	 Intel Core i7-8550U	 AMD Radeon 530 (4 GB)	 2 TB HDD	3149.0	3149.0	0 / 5
HP	ENVY 13-ad004nx	13.3	 Intel Core i7-7500U	 NVIDIA GeForce MX150 (2 GB)	 1 TB PCIe NVMe M.2 SSD	6609.0	6609.0	0 / 5
HP	Pavilion 14-bf102nx	14.0	 Intel Core i5-8250U	 NVIDIA GeForce GT 940MX (2 GB)	 1 TB HDD	3039.0	3039.0	0 / 5
Dell	Inspiron 15 3567	15.6	 Intel Core i7-7500U	 AMD Radeon R5-M340 (2 GB)	 1 TB HDD	2749.0	2749.0	0 / 5
Dell	Inspiron 15 5567	15.6	 Intel Core i7-7500U	 AMD Radeon R7-M445 (4 GB)	 1 TB HDD	3249.0	3249.0	0 / 5
HP	ENVY 13-ad001nx	13.3	 Intel Core i5-7200U	 Intel Graphics 620	 256 GB PCIe NVMe M.2 SSD	3989.0	3989.0	4.1 / 5
HP	ENVY 17-ae003nx	17.3	 Intel Core i7-7500U	 NVIDIA GeForce 940MX (4 GB)	 256 GB NVMe M.2 SSD/1 TB HDD	6199.0	6199.0	0 / 5
HP	15-bs006nx	15.6	 Intel Core i3-6006U	 Intel HD Graphics 520	 1 TB HDD	1779.0	1779.0	3.6 / 5
Dell	XPS 15	15.6	 Intel Core i7-8750H	 NVIDIA GeForce GTX 1050 Ti (4 GB)	 512 GB PCIe NVMe M.2 SSD	8999.0	8999.0	0 / 5
Acer	Aspire A315-51	15.6	 Intel Core i3-6006U	 Intel HD Graphics 520	 500 GB HDD	1649.0	1649.0	3.9 / 5
HP	Pavilion 13-an0000nx	13.3	 Intel Core i5-8265U	 Intel UHD Graphics 620	 128 GB PCIe NVMe M.2 SSD	3149.0	3149.0	0 / 5
Dell	XPS 13 9360	13.3	 Intel Core i5-8250U	 Intel UHD Graphics 620	 256 GB PCIe NVMe M.2 SSD	4399.0	4399.0	0 / 5
Acer	Aspire 5 A515-51G	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX130 (2 GB)	 2 TB HDD	2999.0	2999.0	3.8 / 5
HP	ENVY 13-ad101nx	13.3	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 1 TB PCIe NVMe M.2 SSD	6719.0	6719.0	4.0 / 5
Dell	Inspiron 15 5570	15.6	 Intel Core i5-8250U	 AMD Radeon 530 (2 GB)	 1 TB HDD	2939.0	2939.0	4.0 / 5
Dell	Inspiron 15 3567	15.6	 Intel Core i5-8250U	 AMD Radeon 520 (2 GB)	 1 TB HDD	2549.0	2549.0	0 / 5
Lenovo	IdeaPad 320-15IKBRA	15.6	 Intel Core i5-8250U	 AMD Radeon 530 (2 GB)	 1 TB HDD	2519.0	2519.0	4.2 / 5
Lenovo	IdeaPad 320-14ISK	14.0	 Intel Core i3-6006U	 Intel GMA HD	 1 TB HDD	1779.0	1779.0	0 / 5
Asus	ZenBook UX430UN Ultrabook	14.0	 Intel Core i5-8250U	 NVIDIA GeForce MX150 (2 GB)	 256 GB SSD	3779.0	3779.0	0 / 5
HP	Pavilion 15-cc006nx	15.6	 Intel Core i7-7500U	 NVIDIA GeForce 940MX (2 GB)	 8 GB (Cache Flash)/1 TB HDD	3459.0	3459.0	0 / 5
Dell	Inspiron 15 3567	15.6	 Intel Core i3-6006U	 Intel GMA HD	 1 TB HDD	1889.0	1889.0	0 / 5
Dell	Inspiron 15 3567	15.6	 Intel Core i5-7200U	 AMD Radeon R5-M340 (2 GB)	 1 TB HDD	2469.0	2469.0	0 / 5
HP	ENVY 13-ab000nx	13.3	 Intel Core i5-7200U	 Intel GMA HD	 256 GB PCIe NVMe M.2 SSD	3779.0	3779.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	13.3	 Intel Core i5 Dual Core	 Intel Iris Graphics 550	 256 GB SSD	7768.95	7768.95	0 / 5
HP	\N	15.6	 Intel Core i5-7200U	 AMD Radeon 520 (2 GB)	 1 TB HDD	2419.0	2419.0	0 / 5
HP	Pavilion 15-cc002nx	15.6	 Intel Core i7-7500U	 NVIDIA GeForce 940MX (2 GB)	 8 GB (Cache Flash)/1 TB HDD	3459.0	3459.0	0 / 5
Apple	MacBook Pro (Retina)	13.3	 Intel Core i5 Dual Core	 Intel Iris Plus Graphics 640	 256 GB SSD	6499.0	6499.0	4.6 / 5
HP	ENVY 13-ad100nx	13.3	 Intel Core i7-8550U	 Intel HD Graphics 620	 512 GB NVMe M.2 SSD	4829.0	4829.0	4.0 / 5
Dell	Inspiron 15 3567	15.6	 Intel Core i5-7200U	 AMD Radeon R5-M430 (2 GB)	 1 TB HDD	2249.0	2249.0	3.3 / 5
Lenovo	IdeaPad 320-15IKBRN	15.6	 Intel Core i5-8250U	 Intel GMA HD	 1 TB HDD	2099.0	2099.0	3.8 / 5
Huawei	MateBook D	15.6	 Intel Core i7-8550U	 NVIDIA GeForce MX150 (2 GB)	 128 GB SSD/1 TB HDD	3299.0	3299.0	4.0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i7 6 Core	 Radeon Pro 555X GDDR5 (4 GB)	 256 GB SSD	10199.0	10199.0	0 / 5
Apple	MacBook Pro (Retina + Touch Bar)	15.4	 Intel Core i7 6 Core	 Radeon Pro 560X GDDR5 (4 GB)	 512 GB SSD	11899.0	11899.0	4.4 / 5
Dell	Inspiron 15 3576	15.6	 Intel Core i5-8250U	 AMD Radeon 520 (2 GB)	 1 TB HDD	2299.0	2299.0	0 / 5
\.


--
-- PostgreSQL database dump complete
--


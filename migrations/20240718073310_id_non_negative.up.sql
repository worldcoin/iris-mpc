ALTER TABLE irises
ADD CONSTRAINT id_non_negative CHECK (id >= 0);
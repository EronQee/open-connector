alter table runtime_tokens add column allowed_credential_exports text not null default '[]';

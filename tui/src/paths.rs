use std::ffi::OsString;
use std::path::PathBuf;

/// Root of extty's local data: `$EXTTY_HOME` if set and non-empty, else `~/.extty`.
///
/// Mirrors `extty.storage.get_extty_home` in the Python SDK, including
/// expansion of a leading `~`, so both sides agree on where runs live.
pub fn extty_home() -> PathBuf {
    resolve_extty_home(std::env::var_os("EXTTY_HOME"), dirs::home_dir())
}

fn resolve_extty_home(env: Option<OsString>, home: Option<PathBuf>) -> PathBuf {
    let home = home.unwrap_or_else(|| PathBuf::from("."));
    match env.map(PathBuf::from) {
        Some(dir) if dir.as_os_str().is_empty() => home.join(".extty"),
        Some(dir) => match dir.strip_prefix("~") {
            Ok(rest) => home.join(rest),
            Err(_) => dir,
        },
        None => home.join(".extty"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> Option<PathBuf> {
        Some(PathBuf::from("/home/me"))
    }

    #[test]
    fn defaults_to_dot_extty_in_home() {
        assert_eq!(
            resolve_extty_home(None, home()),
            PathBuf::from("/home/me/.extty")
        );
    }

    #[test]
    fn empty_env_uses_default() {
        assert_eq!(
            resolve_extty_home(Some(OsString::new()), home()),
            PathBuf::from("/home/me/.extty")
        );
    }

    #[test]
    fn env_overrides_default() {
        assert_eq!(
            resolve_extty_home(Some("/scratch/extty".into()), home()),
            PathBuf::from("/scratch/extty")
        );
    }

    #[test]
    fn env_expands_leading_tilde() {
        assert_eq!(
            resolve_extty_home(Some("~/scratch/extty".into()), home()),
            PathBuf::from("/home/me/scratch/extty")
        );
    }
}

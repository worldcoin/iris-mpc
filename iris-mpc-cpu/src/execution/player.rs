use core::num;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Runtime identity of party.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identity(pub String);

impl Default for Identity {
    fn default() -> Self {
        Identity("test_identity".to_string())
    }
}

impl From<&str> for Identity {
    fn from(s: &str) -> Self {
        Identity(s.to_string())
    }
}

impl From<&String> for Identity {
    fn from(s: &String) -> Self {
        Identity(s.clone())
    }
}

impl From<String> for Identity {
    fn from(s: String) -> Self {
        Identity(s)
    }
}

/// Struct that keeps the player id (role), zero indexed;
#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct Role(u8);

impl Role {
    /// Create Role from a 0..N-1 indexing
    pub fn new(x: usize) -> Self {
        Role(x as u8)
    }

    /// Retrieve index of Role (zero indexed)
    pub fn zero_based(&self) -> usize {
        self.0 as usize
    }

    /// Retrieve next player, function used mostly in replicated secret sharing.
    pub fn next(&self, num_players: u8) -> Role {
        Role((self.0 + 1) % num_players)
    }

    /// Retrieve previous player, function used mostly in replicated secret
    /// sharing.
    pub fn prev(&self, num_players: u8) -> Role {
        if self.0 == 0 {
            Role::new(num_players as usize - 1)
        } else {
            Role(self.0 - 1)
        }
    }
}

pub type RoleAssignment = HashMap<Role, Identity>;

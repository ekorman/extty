mod lambda;
mod prime;
mod vast;

use anyhow::Result;

pub use lambda::LambdaProvider;
pub use prime::PrimeProvider;
pub use vast::VastProvider;

use super::models::{Instance, InstanceType, LaunchOptions, Provider};

pub trait CloudProvider: Send {
    fn name(&self) -> Provider;
    fn ssh_user(&self) -> &str;
    fn list_instances(&self) -> Result<Vec<Instance>>;
    fn get_instance(&self, id: &str) -> Result<Instance>;
    fn list_instance_types(&self) -> Result<Vec<InstanceType>>;
    fn launch(&self, opts: &LaunchOptions) -> Result<Vec<String>>;
    fn terminate(&self, ids: &[String]) -> Result<()>;
}

pub fn get_provider(provider: Provider, api_key: &str) -> Box<dyn CloudProvider> {
    match provider {
        Provider::Lambda => Box::new(LambdaProvider::new(api_key)),
        Provider::Vast => Box::new(VastProvider::new(api_key)),
        Provider::Prime => Box::new(PrimeProvider::new(api_key)),
    }
}

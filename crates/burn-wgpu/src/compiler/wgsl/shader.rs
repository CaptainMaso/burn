use super::{Body, Extension, Item};
use burn_jit::{gpu::WorkgroupSize, CompilerRepresentation};
use std::fmt::{Display, Write};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Location {
    Storage,
    Workgroup,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory {
    location: Location,
    pub index: u16,
    item: Item,
    size: u32,
}

impl SharedMemory {
    pub fn new(index: u16, item: Item, size: u32) -> Self {
        Self {
            location: Location::Workgroup,
            index,
            item,
            size,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalArray {
    pub index: u16,
    item: Item,
    name: u8,
    size: u32,
}

impl LocalArray {
    pub fn new(index: u16, item: Item, name: u8, size: u32) -> Self {
        Self {
            index,
            item,
            name,
            size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeShader {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub shared_memories: Vec<SharedMemory>,
    pub local_arrays: Vec<LocalArray>,
    pub workgroup_size: WorkgroupSize,
    pub global_invocation_id: bool,
    pub local_invocation_index: bool,
    pub local_invocation_id: bool,
    pub num_workgroups: bool,
    pub workgroup_id: bool,
    pub features: wgpu::Features,
    pub body: Body,
    pub extensions: Vec<Extension>,
}

impl Display for ComputeShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // for (feat_name, _feat) in self.features.iter_names() {
        //     let feat_name = LowerDash(feat_name);

        //     //writeln!(f, "enable {feat_name};")?;
        // }

        Self::format_bindings(f, "input", &self.inputs, 0)?;
        Self::format_bindings(f, "output", &self.outputs, self.inputs.len())?;

        for (i, (name, binding)) in self.named.iter().enumerate() {
            Self::format_binding(
                f,
                name.as_str(),
                binding,
                self.inputs.len() + self.outputs.len() + i,
            )?;
        }

        for array in self.shared_memories.iter() {
            f.write_fmt(format_args!(
                "var<{}> shared_memory_{}: array<{}, {}>;\n\n",
                array.location, array.index, array.item, array.size
            ))?;
        }

        f.write_fmt(format_args!(
            "const WORKGROUP_SIZE_X = {}u;
const WORKGROUP_SIZE_Y = {}u;
const WORKGROUP_SIZE_Z = {}u;\n",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z
        ))?;

        f.write_fmt(format_args!(
            "
@compute
@workgroup_size({}, {}, {})
fn main(
",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z
        ))?;

        if self.global_invocation_id {
            f.write_str("    @builtin(global_invocation_id) global_id: vec3<u32>,\n")?;
        }

        if self.local_invocation_index {
            f.write_str("    @builtin(local_invocation_index) local_idx: u32,\n")?;
        }

        if self.local_invocation_id {
            f.write_str("    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,\n")?;
        }

        if self.num_workgroups {
            f.write_str("    @builtin(num_workgroups) num_workgroups: vec3<u32>,\n")?;
        }

        if self.workgroup_id {
            f.write_str("    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n")?;
        }

        // Open body
        f.write_fmt(format_args!(") {{"))?;

        // Local arrays
        for array in self.local_arrays.iter() {
            f.write_fmt(format_args!(
                "var a_{}_{}: array<{}, {}>;\n\n",
                array.name, array.index, array.item, array.size
            ))?;
        }

        // Body
        f.write_fmt(format_args!("{}", self.body))?;

        // Close body
        f.write_fmt(format_args!("}}"))?;

        for extension in self.extensions.iter() {
            f.write_fmt(format_args!("{extension}\n\n"))?;
        }

        Ok(())
    }
}

impl ComputeShader {
    fn format_bindings(
        f: &mut core::fmt::Formatter<'_>,
        prefix: &str,
        bindings: &[Binding],
        num_entry: usize,
    ) -> core::fmt::Result {
        for (i, binding) in bindings.iter().enumerate() {
            Self::format_binding(
                f,
                format!("{prefix}_{i}_global").as_str(),
                binding,
                num_entry + i,
            )?;
        }

        Ok(())
    }

    fn format_binding(
        f: &mut core::fmt::Formatter<'_>,
        name: &str,
        binding: &Binding,
        num_entry: usize,
    ) -> core::fmt::Result {
        let ty = match binding.size {
            Some(size) => format!("array<{}, {}>", binding.item, size),
            None => format!("array<{}>", binding.item),
        };

        f.write_fmt(format_args!(
            "@group(0)
@binding({})
var<{}, {}> {}: {};
\n",
            num_entry, binding.location, binding.visibility, name, ty
        ))?;

        Ok(())
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Location::Storage => f.write_str("storage"),
            Location::Workgroup => f.write_str("workgroup"),
        }
    }
}

impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Read => f.write_str("read"),
            Visibility::ReadWrite => f.write_str("read_write"),
        }
    }
}

impl CompilerRepresentation for ComputeShader {
    fn shared_memory_size(&self) -> usize {
        // not used in wgsl compiler
        0
    }
}

struct LowerDash<'a>(&'a str);

impl LowerDash<'_> {
    fn write_lower(s: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in s.split_inclusive(|c: char| c.is_uppercase()) {
            let Some(last_char) = s.chars().last() else {
                continue;
            };
            let s = &s[..s.len() - last_char.len_utf8()];
            f.write_str(s)?;
            write!(f, "{}", last_char.to_lowercase())?;
        }
        Ok(())
    }
}

impl Display for LowerDash<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self
            .0
            .split_whitespace()
            .flat_map(|s| s.split_terminator('_'));

        if let Some(s) = iter.next() {
            Self::write_lower(s, f)?;
        } else {
            return Ok(());
        }

        for s in iter {
            f.write_char('-')?;
            Self::write_lower(s, f)?;
        }

        Ok(())
    }
}

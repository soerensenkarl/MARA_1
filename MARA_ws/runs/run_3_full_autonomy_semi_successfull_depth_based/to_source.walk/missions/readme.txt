Missions have multiple formats.

Files without an extension should be serialized bosdyn.api.mission.Node protobufs.  This is the older
format.  These files can be deserialized and sent directly to the mission service.

Files with the '.walk' extension should be serialized bosdyn.api.autowalk.Walk protobufs. This is the newer
format.  These files need to be converted to bosdyn.api.mission.Node protobufs before sending to the
mission service.  This can be done using the autowalk service.

Files with the '.node' extension should be serialized bosdyn.api.mission.Node protobufs.

As of 3.3, we NO LONGER save '.node' files for autowalk missions.

If there are files with the same name, but a different extension, the tablet will first try to load '.walk'
files, then '.node' files, then extensionless files.  If the tablet can convert the older, extensionless format
to the new '.walk' format it will.
